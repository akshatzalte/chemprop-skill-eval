"""Run a command inside a bubblewrap box that can use GPUs.

Claude Code's own Bash sandbox replaces /dev, so CUDA is unavailable inside it. For GPU tasks the
whole `claude -p` process runs in this box instead, with Claude Code's sandbox turned off. Inside
the box only these paths exist:

- the run directory (read-write; it holds the working directory and a throwaway HOME)
- /usr, /etc, /sys and the root symlinks into /usr (read-only)
- the Python env and the Claude Code installation (read-only)
- the GPU device files for the GPUs the task may use

The home directory, other runs, /data and the rest of /tmp do not exist in the box. Claude Code
reaches the API over the network, so the network is shared with the host. The user's login is not
copied in: Claude Code gets only a short-lived OAuth access token through the environment, and no
refresh token, so nothing in the box can renew or replace the user's credentials.
"""

import json
import os
from pathlib import Path
import time

HOME = Path.home()
CLAUDE_INSTALL = HOME / ".local" / "share" / "claude"
CREDENTIALS = HOME / ".claude" / ".credentials.json"


def access_token(min_remaining_s: int) -> str:
    """The current OAuth access token, if it stays valid for at least min_remaining_s."""
    oauth = json.loads(CREDENTIALS.read_text())["claudeAiOauth"]
    remaining = oauth["expiresAt"] / 1000 - time.time()
    if remaining < min_remaining_s:
        raise RuntimeError(
            f"access token expires in {remaining / 60:.0f} min, before a run could finish; retry "
            "after Claude Code renews it (it does so when the token is about to expire)"
        )
    return oauth["accessToken"]


def gpu_devices(visible: str) -> list[str]:
    devices = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"]
    devices += [f"/dev/nvidia{i.strip()}" for i in visible.split(",") if i.strip()]
    return [d for d in devices if os.path.exists(d)]


def bwrap_command(
    cmd: list[str],
    run_dir: Path,
    work: Path,
    python_env: Path,
    gpus: str,
    extra_ro: list[tuple[Path, Path]],
    min_token_s: int,
) -> tuple[list[str], dict]:
    """bwrap argv wrapping cmd, and the environment variables the box needs."""
    home = run_dir / "home"
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    args = [
        "bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc", "--ro-bind", "/sys", "/sys",
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        # /etc/resolv.conf points here on Ubuntu (systemd-resolved); needed to reach the API
        "--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve",
        "--ro-bind", str(python_env), str(python_env),
        "--ro-bind", str(CLAUDE_INSTALL), str(CLAUDE_INSTALL),
        "--bind", str(run_dir), str(run_dir),
        "--chdir", str(work),
    ]  # fmt: skip
    for device in gpu_devices(gpus):
        args += ["--dev-bind", device, device]
    for source, destination in extra_ro:
        args += ["--ro-bind", str(source), str(destination)]
    env = {
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "CLAUDE_CODE_OAUTH_TOKEN": access_token(min_token_s),
        "TMPDIR": "/tmp",
        # only the allowed GPUs' device files exist in the box, and the driver numbers them from 0
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(len(gpus.split(",")))),
    }
    return args + ["--", *cmd], env
