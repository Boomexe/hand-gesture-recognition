"""Record and apply the checkpoint-I/O repair to an existing failed frozen suite.

Retains the original plan, original code, all results, and the failed attempt.
Only orchestration/checkpoint-I/O sources are replaced; data and training settings
remain unchanged. Run --execute --resume afterward to continue the queue.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite_dir.resolve()
    status = json.loads((suite/"status.json").read_text())
    if status["status"] != "failed":
        raise ValueError("Only a failed suite may be repaired")
    plan_path = suite/"plan.json"
    plan = json.loads(plan_path.read_text())
    for relative, digest in plan["source_sha256"].items():
        if hashlib.sha256((suite/"code"/relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source differs from its plan: {relative}")
    archive = suite/"revisions/checkpoint_io_repair"
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan_path, archive/"original_plan.json")
    shutil.copy2(suite/"status.json", archive/"failure_status.json")
    for name in ("suite.stdout.log", "suite.stderr.log"):
        if (suite/name).exists():
            shutil.copy2(suite/name, archive/name)
    changes = []
    for relative in ("experiments/train_features.py", "experiments/run_training_suite.py", "experiments/checkpoints.py"):
        target = suite/"code"/relative
        before = None
        if target.exists():
            original = archive/relative
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, original)
            before = hashlib.sha256(target.read_bytes()).hexdigest()
        shutil.copy2(ROOT/relative, target)
        after = hashlib.sha256(target.read_bytes()).hexdigest()
        changes.append({"file": relative, "before_sha256": before, "after_sha256": after})
        plan["source_sha256"][relative] = after
    note = {"at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Windows error 1224 while overwriting best.pt during the second run",
            "changes": changes,
            "protocol_effect": "Checkpoint storage and queue recovery only; features, architecture, seeds, splits, and hyperparameters unchanged",
            "failed_attempt_policy": "Preserve and restart incomplete training in a new retry directory; no optimizer state was saved",
            "completed_runs_preserved": status["completed_runs"]}
    plan.setdefault("amendments", []).append(note)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (archive/"amendment.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"Recorded checkpoint-I/O repair; completed runs preserved: {status['completed_runs']}")


if __name__ == "__main__":
    main()
