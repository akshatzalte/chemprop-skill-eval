"""Re-grade finished runs from the files and traces they left, without re-running any agents.

usage: python scripts/regrade.py results/<name>

Use this after fixing a grader. It rewrites results/<name>/runs.jsonl in place (keeping a backup)
and prints the runs whose result changed.
"""

import argparse
import json
from pathlib import Path
import shutil

from grade import grade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    path = args.results / "runs.jsonl"
    shutil.copy(path, path.with_suffix(".jsonl.bak"))
    records = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        run_dir = Path(record["run_dir"])
        new = grade(record["task"], run_dir / "work", run_dir / "trace.jsonl")
        if new["passed"] != record["passed"]:
            run = f"{record['model']} {record['task']} {record['arm']}"
            print(f"{run}: {record['passed']} -> {new['passed']}")
        records.append({**record, **new})
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


if __name__ == "__main__":
    main()
