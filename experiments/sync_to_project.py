"""Copy scoped worktree changes and results into an existing project safely.

Never deletes target files. Conflicting target content is backed up. A bounded
training suite can finish in the source worktree before a final verified sync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.json_io import atomic_write_json

TOP_FILES = (
    "detect_gestures.ipynb", "gesture_model.py", "parameters.json",
    "website/app.py", "website/live_cache_debug.csv", "website/parameters.json",
    "website/parameters_v10_full_dictionary.json", "JHSS_PROJECT_ISSUES.md",
    "JHSS_DEVELOPMENT_RECORD.md", "tests/test_prefix_evaluation.py",
    "tests/test_experiment_checkpoints.py", "tests/test_experiment_json_io.py",
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def source_files():
    files = [ROOT/name for name in TOP_FILES if (ROOT/name).is_file()]
    files.extend(path for path in (ROOT/"experiments").rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts
                 and path.suffix not in (".pyc", ".tmp"))
    return sorted(files)


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="merge-copy-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        before = source.stat()
        shutil.copy2(source, temporary)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return None
        copied_hash = digest(temporary)
        for attempt in range(10):
            try:
                temporary.replace(destination)
                return copied_hash
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(min(0.05*2**attempt, 0.5))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def suite_status(suite):
    snapshots = sorted((suite/"status_history").glob("*.json"))
    status_path = snapshots[-1] if snapshots else suite/"status.json"
    return json.loads(status_path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--backup-name", required=True)
    parser.add_argument("--wait-suite", type=Path)
    args = parser.parse_args()
    target = args.target.resolve()
    if not target.is_dir() or target == ROOT or target.is_relative_to(ROOT) or ROOT.is_relative_to(target):
        raise ValueError("Source and existing target must be separate directories")
    backup = target/"merge_backups"/args.backup_name
    backup.mkdir(parents=True, exist_ok=False)
    record = {"source": str(ROOT), "target": str(target), "backup": str(backup),
              "copied": {}, "preserved_line_endings": [], "backups": [], "deferred": []}

    def sync():
        deferred = []
        copied = 0
        for source in source_files():
            relative = source.relative_to(ROOT).as_posix()
            destination = target/relative
            source_hash = digest(source)
            if destination.exists():
                target_hash = digest(destination)
                if source_hash == target_hash:
                    continue
                if relative == "detect_gestures.ipynb" and source.read_text() == destination.read_text():
                    if relative not in record["preserved_line_endings"]:
                        record["preserved_line_endings"].append(relative)
                    continue
                # Do not back up our own previous sync of a changing result file.
                if record["copied"].get(relative) != target_hash:
                    original = backup/relative
                    if original.exists():
                        raise RuntimeError(f"Target changed again after backup: {relative}")
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, original)
                    record["backups"].append(relative)
            copied_hash = atomic_copy(source, destination)
            if copied_hash is None:
                deferred.append(relative)
                continue
            record["copied"][relative] = copied_hash
            copied += 1
        record["deferred"] = deferred
        atomic_write_json(backup/"merge_report.json", record)
        print(f"Sync: {copied} files copied/updated; {len(deferred)} active writes deferred; "
              f"{len(record['backups'])} target conflicts backed up", flush=True)

    sync()
    if args.wait_suite:
        previous = None
        while True:
            status = suite_status(args.wait_suite)
            summary = (status["status"], len(status["completed_runs"]), status.get("current_run"))
            if summary != previous:
                print(f"Training: status={summary[0]}; completed={summary[1]}/{status['total_runs']}; "
                      f"current={summary[2]}", flush=True)
                previous = summary
            if status["status"] in ("complete", "failed"):
                sync()
                record["training_status_at_final_sync"] = status["status"]
                break
            time.sleep(10)
    else:
        sync()
    mismatches = []
    for source in source_files():
        relative = source.relative_to(ROOT).as_posix()
        destination = target/relative
        if not destination.is_file():
            mismatches.append(relative)
        elif relative == "detect_gestures.ipynb" and source.read_text() == destination.read_text():
            continue
        elif digest(source) != digest(destination):
            mismatches.append(relative)
    record["verification_mismatches"] = mismatches
    record["verified_file_count"] = len(source_files())
    atomic_write_json(backup/"merge_report.json", record)
    if mismatches:
        raise RuntimeError(f"Final copy verification failed: {mismatches}")
    print(f"Verified {record['verified_file_count']} merged files in {target}", flush=True)


if __name__ == "__main__":
    main()
