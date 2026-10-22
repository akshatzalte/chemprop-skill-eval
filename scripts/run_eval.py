"""Run each task with and without the Chemprop agent skill using headless Claude Code (`claude -p`).

usage:
    python scripts/run_eval.py --name pilot --models haiku sonnet --runs 1 --budget-usd 25

Every run starts from a fresh copy of .workspace (built by scripts/build_workspace.sh) in its own
directory under --runs-root. In the "with" arm the frozen skill in skill/chemprop is copied to
<work>/.claude/skills/chemprop; that is the only difference between the arms.

Isolation and safety (see settings_for below):
- user-level Claude Code settings are not loaded (--setting-sources project), so personal plugins,
  hooks, model and effort defaults do not apply; MCP servers and claude.ai skill sync are off
- shell commands run in Claude Code's sandbox (bubblewrap) with no network access, writes limited
  to the run directory, and reads of the home directory blocked except the Python env, the
  CheMeleon cache, and the run directory
- file tools may only write inside the run directory and cannot read the home directory,
  /data, or the session scratch space outside it
Results are appended to results/<name>/runs.jsonl; traces are kept next to each run.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

from box import bwrap_command
from grade import grade

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
PYTHON_ENV = HOME / "anaconda3" / "envs" / "chemprop_skill_eval"
# skill copied into <work>/.claude/skills/chemprop in each arm ("without" gets none)
SKILL_SOURCES = {
    "with": Path("skill/chemprop"),
    "with_short": Path("skill_short/chemprop"),
    "with_pr": Path("skill_pr/chemprop"),  # the current PR version (short + CheMeleon rewrite)
}
CLAUDE = str(Path(shutil.which("claude") or HOME / ".local" / "bin" / "claude").resolve())
# GPUs visible to each task (CUDA_VISIBLE_DEVICES); all other tasks run on CPU. GPU 0 is shared
# with another user's long-running job, so single-GPU tasks use GPU 1.
GPU_TASKS = {"gpu_train": "1", "ddp_train": "0,1", "gpu_check": "1", "isolation_box": "1"}
TOOLS = ["Bash", "Read", "Glob", "Grep", "Edit", "Write", "Skill", "TodoWrite"]
# /tmp covers other runs (which may hold a copy of the skill) and the Claude Code session scratch
DENIED_PATHS = [str(HOME), "/data", "/tmp"]
# Each run lives in a random directory directly inside this one, which has no read permission: an
# agent knows its own path but cannot list the parent, so it cannot find other runs (or their skill
# copies). Per user, so another user's directory of the same name can't get in the way.
RUNS_PARENT = Path(f"/tmp/cse-runs-{os.environ.get('USER', 'user')}")
# for the file tools (shell commands are covered by the sandbox's denyRead)
DENIED_FOR_FILE_TOOLS = [str(HOME), "/data", "/tmp/claude-1017", "/tmp/chemprop-skill-eval-runs"]


def settings_for(work: Path, tmp: Path) -> dict:
    deny = [f"{tool}(/{p}/**)" for p in DENIED_FOR_FILE_TOOLS for tool in ("Read", "Edit", "Write")]
    allow = [f"{tool}(/{work}/**)" for tool in ("Read", "Edit", "Write")]
    allow += ["Bash", "Glob", "Grep", "Skill", "TodoWrite"]
    return {
        "permissions": {
            "allow": allow,
            "deny": deny,
            "defaultMode": "dontAsk",
            # not blockReadsOutsideWorkingDirectories: it also refuses every shell command whose
            # variables it cannot check in advance, which in round 2 blocked a few legitimate commands
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "denyRead": DENIED_PATHS,
                "allowRead": [str(work), str(tmp), str(PYTHON_ENV), str(HOME / ".chemprop")],
                "denyWrite": [str(HOME), "/data"],
                "allowWrite": [str(work), str(tmp)],
            },
            "network": {"allowedDomains": [], "strictAllowlist": True},
        },
        "syncClaudeAiSkills": False,
        "syncClaudeAiPlugins": False,
        "disableClaudeAiConnectors": True,
        "enableArtifact": False,
    }


def trace_events(trace: Path) -> list[dict]:
    """The JSON events of a trace, skipping any line that isn't complete JSON (e.g. after a kill)."""
    events = []
    for line in trace.read_text(errors="replace").splitlines() if trace.is_file() else []:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def tasks_in(directory: Path) -> list[str]:
    """Task prompts in a directory; upper-case names (README, ROUND3) are documentation."""
    return sorted(p.stem for p in directory.glob("*.md") if not p.stem.isupper())


def check_isolation(trace: Path, arm: str) -> list[str]:
    """Problems with what the session loaded, from the init event of the trace."""
    for event in trace_events(trace):
        if event.get("type") == "system" and event.get("subtype") == "init":
            skills = [str(s) for s in event.get("skills", [])]
            plugins = [str(p.get("name", p)) for p in event.get("plugins", [])]
            problems = []
            expected = arm != "without"
            if expected != any(s.endswith("chemprop") for s in skills):
                problems.append(f"chemprop skill loaded={expected} expected, got skills={skills}")
            if any("superpowers" in p for p in plugins + skills):
                problems.append(f"user plugin leaked: plugins={plugins}")
            return problems
    return ["no init event in trace"]


def summarize_trace(trace: Path) -> dict:
    info = {"skill_invoked": False, "skill_files_read": 0, "repo_files_read": 0}
    result = {}
    for line in trace.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []):
            if block.get("type") != "tool_use":
                continue
            tool_input = block.get("input") or {}
            if block["name"] == "Skill" and "chemprop" in str(tool_input.get("skill", "")):
                info["skill_invoked"] = True
            elif block["name"] == "Read":
                path = str(tool_input.get("file_path", ""))
                info["skill_files_read"] += "/.claude/skills/chemprop/" in path
                info["repo_files_read"] += "/work/chemprop/" in path
    usage = result.get("usage") or {}
    return {
        **info,
        "is_error": result.get("is_error"),
        "subtype": result.get("subtype"),
        "num_turns": result.get("num_turns"),
        "duration_s": (result.get("duration_ms") or 0) / 1000,
        "cost_usd": result.get("total_cost_usd"),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "model_usage": result.get("modelUsage"),
        "final_message": str(result.get("result", ""))[:2000],
    }


def run_one(args, model: str, task: str, arm: str, i: int) -> dict:
    run_dir = args.runs_root / uuid.uuid4().hex
    RUNS_PARENT.chmod(0o311)  # keep it unlistable even if something reset the mode
    work = run_dir / "work"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"model": model, "task": task, "arm": arm, "run": i})
    )
    shutil.copytree(ROOT / args.workspace, work, symlinks=True)
    if arm != "without":
        shutil.copytree(ROOT / SKILL_SOURCES[arm], work / ".claude" / "skills" / "chemprop")
    # Claude Code puts the sandbox's per-user temp directory (and its proxy sockets) under $TMPDIR;
    # keep the path short, since Unix socket paths are limited to ~108 characters
    tmp = Path(tempfile.mkdtemp(prefix="cse-", dir="/tmp"))
    boxed = task in GPU_TASKS  # GPU tasks run in our own box (see box.py) instead of the sandbox
    settings = run_dir / "settings.json"
    run_settings = settings_for(work, tmp)
    if boxed:
        # inside the box nothing outside the run directory exists, so the box replaces both the
        # sandbox and the read block (which also refuses any command whose variables it can't check)
        run_settings["sandbox"] = {"enabled": False}
        run_settings["permissions"]["blockReadsOutsideWorkingDirectories"] = False
    settings.write_text(json.dumps(run_settings, indent=2))
    prompt = (args.tasks_dir / f"{task}.md").read_text()

    cmd = [
        CLAUDE, "-p", prompt,
        "--model", model,
        "--output-format", "stream-json", "--verbose",
        "--max-turns", str(args.max_turns),
        "--max-budget-usd", str(args.per_run_budget_usd),
        "--permission-mode", "dontAsk",
        "--setting-sources", "project",
        "--settings", str(settings),
        "--strict-mcp-config",
        "--no-session-persistence",
        "--tools", ",".join(TOOLS),
    ]  # fmt: skip
    env = {
        "HOME": str(HOME),
        "USER": os.environ.get("USER", ""),
        "LOGNAME": os.environ.get("LOGNAME", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "SHELL": "/bin/bash",
        "PATH": f"{PYTHON_ENV / 'bin'}:/usr/local/bin:/usr/bin:/bin",
        "CUDA_VISIBLE_DEVICES": GPU_TASKS.get(task, ""),
        "TMPDIR": str(tmp),
        # be a good neighbour on a shared machine; same limit in both arms
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    }
    if boxed:
        # the box enforces isolation, so commands are not gated one by one inside it
        cmd[cmd.index("dontAsk")] = "bypassPermissions"
        cmd, box_env = bwrap_command(
            cmd,
            run_dir,
            work,
            PYTHON_ENV,
            GPU_TASKS[task],
            extra_ro=[(HOME / ".chemprop", run_dir / "home" / ".chemprop")],
            min_token_s=args.timeout_s + 300,
        )
        env.update(box_env)
    trace = run_dir / "trace.jsonl"
    started = time.time()
    with open(trace, "w") as out, open(run_dir / "stderr.log", "w") as err:
        try:
            proc = subprocess.run(
                cmd, cwd=work, env=env, stdout=out, stderr=err, timeout=args.timeout_s
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code = "timeout"

    try:
        record = {
            "model": model,
            "task": task,
            "arm": arm,
            "run": i,
            "exit_code": exit_code,
            "wall_s": round(time.time() - started, 1),
            "isolation_problems": check_isolation(trace, arm),
            **summarize_trace(trace),
            **grade(task, work, trace),
            "run_dir": str(run_dir),
            "boxed": boxed,
        }
    finally:
        # Keep what the agent produced; drop the skill copy (so other runs can never find it) and
        # the large unchanged copies of the repository and the data (graders use .workspace*).
        for big in ("chemprop", "data", ".claude/skills"):
            shutil.rmtree(work / big, ignore_errors=True)
        # Lightning checkpoints and training logs can take gigabytes per run (round 3 filled the
        # shared disk); the final model (best.pt / model.pt) is kept for re-grading
        for heavy in list(work.rglob("checkpoints")) + list(work.rglob("*_logs")):
            shutil.rmtree(heavy, ignore_errors=True)
        for ckpt in work.rglob("*.ckpt"):
            ckpt.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="results/<name>/")
    parser.add_argument("--models", nargs="+", default=["haiku"])
    parser.add_argument("--tasks-dir", type=Path, default=ROOT / "tasks")
    parser.add_argument("--tasks", nargs="+", help="default: every task in --tasks-dir")
    parser.add_argument(
        "--workspace",
        default=".workspace",
        help="directory every run starts from (round 3: .workspace_r3)",
    )
    parser.add_argument(
        "--arms", nargs="+", default=["without", "with"], choices=["without", *SKILL_SOURCES]
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run-offset", type=int, default=0, help="number runs from this index")
    parser.add_argument(
        "--budget-usd", type=float, required=True, help="stop launching runs past this"
    )
    parser.add_argument("--per-run-budget-usd", type=float, default=3.0)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--resume", action="store_true", help="skip runs already in results/<name>/runs.jsonl"
    )
    parser.add_argument(
        "--min-free-gb", type=float, default=15, help="don't start a run with less free disk"
    )
    parser.add_argument("--runs-root", type=Path, default=None)
    args = parser.parse_args()
    out_dir = ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    args.tasks = args.tasks or tasks_in(args.tasks_dir)
    if args.runs_root is None:
        RUNS_PARENT.mkdir(exist_ok=True)
        RUNS_PARENT.chmod(0o311)
        args.runs_root = RUNS_PARENT
        print(f"run directories: random names in {RUNS_PARENT} (see run_dir in runs.jsonl)")

    # interleave arms so both see the same conditions (time of day, rate limits)
    jobs = [
        (m, t, a, i)
        for m in args.models
        for i in range(args.run_offset, args.run_offset + args.runs)
        for t in args.tasks
        for a in args.arms
    ]
    # --resume: skip runs already recorded, and count what they cost against the budget
    done, already_spent = set(), 0.0
    if args.resume and (out_dir / "runs.jsonl").is_file():
        for line in (out_dir / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["task"], r["arm"], r["run"]))
            already_spent += r.get("cost_usd") or 0
        jobs = [j for j in jobs if j not in done]
        print(f"resuming: {len(done)} runs already done (${already_spent:.2f}), {len(jobs)} to go")
    lock, spent = threading.Lock(), [already_spent]

    def job(spec):
        with lock:
            if spent[0] >= args.budget_usd:
                return None
            # the disk is shared with other users: never be the ones who fill it
            free_gb = shutil.disk_usage(args.runs_root).free / 1e9
            if free_gb < args.min_free_gb:
                print(
                    f"skipped {spec}: only {free_gb:.1f} GB free (< {args.min_free_gb})", flush=True
                )
                return None
        record = run_one(args, *spec)
        with lock:
            spent[0] += record.get("cost_usd") or 0
            with open(out_dir / "runs.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"[{spent[0]:7.2f} $] {record['model']:>7} {record['task']:<22} {record['arm']:<7} "
                f"passed={record['passed']!s:<5} turns={record['num_turns']} "
                f"cost={record['cost_usd']} isolation={record['isolation_problems'] or 'ok'}",
                flush=True,
            )
        return record

    with ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(job, jobs))
    print(f"done: {spent[0]:.2f} $ spent; results in {out_dir / 'runs.jsonl'}")


if __name__ == "__main__":
    main()
