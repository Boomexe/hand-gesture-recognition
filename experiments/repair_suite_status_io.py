"""Record the status-file sharing-lock repair without changing the experiment plan."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.json_io import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite_dir.resolve()
    status = json.loads((suite/"status.json").read_text())
    if status["status"] != "failed":
        raise ValueError("Only a failed queue may be repaired")
    plan = json.loads((suite/"plan.json").read_text())
    for relative, digest in plan["source_sha256"].items():
        if hashlib.sha256((suite/"code"/relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected frozen-source change: {relative}")
    archive = suite/"revisions/status_io_repair"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("plan.json", "status.json", "suite.stdout.log", "suite.stderr.log"):
        if (suite/name).exists():
            shutil.copy2(suite/name, archive/name)
    changes = []
    for relative in ("experiments/run_training_suite.py", "experiments/checkpoints.py", "experiments/json_io.py"):
        target = suite/"code"/relative
        before = None
        if target.exists():
            old = archive/relative
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
            before = hashlib.sha256(target.read_bytes()).hexdigest()
        shutil.copy2(ROOT/relative, target)
        after = hashlib.sha256(target.read_bytes()).hexdigest()
        plan["source_sha256"][relative] = after
        changes.append({"file": relative, "before_sha256": before, "after_sha256": after})
    amendment = {"at_utc": datetime.now(timezone.utc).isoformat(),
                 "reason": "WinError 5 replacing status.json after seven completed runs",
                 "changes": changes, "completed_runs_preserved": status["completed_runs"],
                 "protocol_effect": "JSON publication and advisory status handling only; all model and training settings unchanged"}
    plan.setdefault("amendments", []).append(amendment)
    atomic_write_json(suite/"plan.json", plan)
    atomic_write_json(archive/"amendment.json", amendment)
    print(f"Recorded status-I/O repair; preserving {len(status['completed_runs'])} completed runs", flush=True)


if __name__ == "__main__":
    main()
