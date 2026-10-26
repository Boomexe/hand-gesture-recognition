"""Prepare and execute the fixed 18-run, validation-only ablation suite.

Preparation freezes code and labels. Execution uses one GPU process at a time;
no test-set evaluation, model deployment, or notebook changes are performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.json_io import atomic_write_json

SOURCE_FILES = (
    "gesture_model.py", "gesture_preprocessing.py",
    "experiments/features.py", "experiments/train_features.py",
    "experiments/evaluate_prefixes.py", "experiments/run_training_suite.py",
    "experiments/checkpoints.py",
    "experiments/json_io.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    atomic_write_json(path, value)


def publish_status(suite, status):
    history = suite/"status_history"
    history.mkdir(exist_ok=True)
    snapshot = history/f"{time.time_ns():020d}.json"
    write_json(snapshot, status)
    try:
        write_json(suite/"status.json", status)
    except OSError as error:
        if not isinstance(error, PermissionError) and getattr(error, "winerror", None) not in (5, 32, 33, 1224):
            raise
        # The immutable snapshot is authoritative. A locked UI status file must
        # not stop training or incorrectly mark the experiment as failed.
        print(f"Status mirror temporarily locked; saved {snapshot}: {error}", file=sys.stderr, flush=True)


def read_status(suite):
    history = suite/"status_history"
    snapshots = sorted(history.glob("*.json")) if history.exists() else []
    path = snapshots[-1] if snapshots else suite/"status.json"
    return json.loads(path.read_text())


def prepare(args):
    suite = args.suite_dir.resolve()
    if suite.exists():
        raise ValueError("Suite directory already exists; execution uses --execute")
    if not args.data_dir or not args.parameters:
        raise ValueError("Preparation requires --data-dir and --parameters")
    parameters = json.loads(args.parameters.read_text())
    mapping = parameters["numbered_unique_words"]
    if len(mapping) != 52 or sorted(mapping.values()) != list(range(52)):
        raise ValueError("This prespecified suite requires the existing contiguous 52-class vocabulary")
    data = args.data_dir.resolve()
    data_files = {}
    for name in ("cleaned_train_data.h5", "cleaned_val_data.h5"):
        path = data/name
        info = path.stat()
        data_files[name] = {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    suite.mkdir(parents=True)
    snapshot = suite/"code"
    hashes = {}
    for relative in SOURCE_FILES:
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(suite/"parameters.json", parameters)
    runs = []
    for seed in (42, 17, 73):
        # Run the primary matched pair first, then the supporting feature comparisons.
        for variant, prefix in (("positions", 0.0), ("positions", 0.5),
                                ("scaled-positions", 0.0), ("positions-diff", 0.0),
                                ("positions-motion", 0.0), ("legacy", 0.0)):
            name = f"{variant}_{'prefix' if prefix else 'full'}_s{seed}"
            runs.append({"name": name, "variant": variant,
                         "prefix_probability": prefix, "seed": seed})
    plan = {"prepared_at_utc": utc_now(), "python": str(Path(sys.executable).resolve()),
            "data_dir": str(data), "data_files": data_files, "source_sha256": hashes,
            "parameters_sha256": hashlib.sha256((suite/"parameters.json").read_bytes()).hexdigest(),
            "class_count": len(mapping), "runs": runs,
            "training": {"epochs": 150, "patience": 15, "batch_size": 32,
                         "learning_rate": 0.001, "device": "cuda"},
            "evaluation": {"split": "validation", "fractions": [i/10 for i in range(1, 11)]},
            "notes": ["All runs start from scratch; no continuation from the deployed checkpoint.",
                      "All selected training and validation clips are used; no sample limit.",
                      "Best complete-sequence validation loss selects the checkpoint in every condition.",
                      "Seeds 42, 17, 73; one active GPU training/evaluation process.",
                      "Test set is not loaded or scored by this suite.",
                      "Feature dimensions change parameter count, recorded in each run.",
                      "The existing cached data has incompletely recovered signer provenance."]}
    write_json(suite/"plan.json", plan)
    write_json(suite/"status.json", {"status": "prepared", "updated_at_utc": utc_now(),
                                    "completed_runs": [], "total_runs": len(runs)})
    print(f"Prepared {len(runs)} runs in {suite}", flush=True)


def next_available_directory(base):
    candidate = base
    attempt = 1
    while candidate.exists():
        candidate = base.with_name(base.name+f"_retry{attempt}")
        attempt += 1
    return candidate


def training_directory(base):
    candidate = base
    attempt = 1
    while candidate.exists():
        if (candidate/"training_complete.json").is_file():
            return candidate, True
        candidate = base.with_name(base.name+f"_retry{attempt}")
        attempt += 1
    return candidate, False


def execute(suite, resume=False):
    suite = suite.resolve()
    plan = json.loads((suite/"plan.json").read_text())
    status = read_status(suite)
    expected_status = "failed" if resume else "prepared"
    if status["status"] != expected_status:
        raise ValueError(f"Suite is {status['status']}; refusing duplicate execution")
    for name, digest in plan["source_sha256"].items():
        if hashlib.sha256((suite/"code"/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {name}")
    if hashlib.sha256((suite/"parameters.json").read_bytes()).hexdigest() != plan["parameters_sha256"]:
        raise ValueError("Frozen parameters changed")
    for value in plan["data_files"].values():
        info = Path(value["path"]).stat()
        if (info.st_size, info.st_mtime_ns) != (value["bytes"], value["mtime_ns"]):
            raise ValueError(f"Dataset changed: {value['path']}")
    comparison_path = suite/"validation_comparison.json"
    results = json.loads(comparison_path.read_text()) if resume and comparison_path.exists() else []
    result_names = [row["name"] for row in results]
    if len(set(result_names)) != len(result_names) or set(result_names) != set(status["completed_runs"]):
        raise ValueError("Completed result records disagree with queue status")
    for result in results:
        if not Path(result["checkpoint"]).is_file() or not (Path(result["validation_results"])/"summary.json").is_file():
            raise ValueError(f"Completed result artifact is missing: {result['name']}")
    status.pop("error", None)
    status.update(status="running", runner_pid=__import__("os").getpid())
    if resume:
        status["resumed_at_utc"] = utc_now()
    else:
        status["started_at_utc"] = utc_now()

    def publish(**fields):
        status.update(fields, updated_at_utc=utc_now())
        publish_status(suite, status)

    def run_command(command, log_path):
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=str(suite/"code"), stdout=log,
                                       stderr=subprocess.STDOUT)
            publish(child_pid=process.pid, current_log=str(log_path))
            try:
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait(timeout=30)
                raise
        if code:
            raise RuntimeError(f"Command exited {code}; see {log_path}")

    try:
        publish()
        # Imported here so preparation can still run using only the standard library.
        sys.path.insert(0, str(suite/"code"))
        from experiments.checkpoints import resolve_best_checkpoint

        for index, run in enumerate(plan["runs"], 1):
            name = run["name"]
            if name in result_names:
                print(f"[{index}/{len(plan['runs'])}] Preserving completed run {name}", flush=True)
                continue
            out, reuse_training = training_directory(suite/"runs"/name)
            evaluation = next_available_directory(suite/"validation"/out.name)
            log_dir = suite/"logs"
            log_dir.mkdir(exist_ok=True)
            publish(current_run=name, run_number=index, stage="training", current_run_directory=str(out))
            print(f"[{index}/{len(plan['runs'])}] Training {name}", flush=True)
            command = [plan["python"], "-u", str(suite/"code/experiments/train_features.py"),
                       "--data-dir", plan["data_dir"], "--parameters", str(suite/"parameters.json"),
                       "--output", str(out), "--variant", run["variant"],
                       "--prefix-probability", str(run["prefix_probability"]), "--seed", str(run["seed"])]
            for key, value in plan["training"].items():
                command.extend(["--"+key.replace("_", "-"), str(value)])
            started = time.perf_counter()
            if not reuse_training:
                run_command(command, log_dir/(out.name+"_train.log"))
            checkpoint = resolve_best_checkpoint(out)
            publish(stage="validation")
            run_command([plan["python"], "-u", str(suite/"code/experiments/evaluate_prefixes.py"),
                         "--checkpoint", str(checkpoint),
                         "--data", str(Path(plan["data_dir"])/"cleaned_val_data.h5"),
                         "--parameters", str(suite/"parameters.json"), "--output", str(evaluation),
                         "--device", "cuda"], log_dir/(evaluation.name+"_validation.log"))
            history = json.loads((out/"history.json").read_text())
            summary = json.loads((evaluation/"summary.json").read_text())
            best = min(history, key=lambda row: row["validation_loss"])
            full = next(row for row in summary if row["fraction"] == 1)
            half = next(row for row in summary if row["fraction"] == 0.5)
            result = {**run, "epochs_run": len(history), "best_epoch": best["epoch"],
                      "best_validation_loss": best["validation_loss"],
                      "full_validation_accuracy": full["accuracy"],
                      "full_validation_macro_f1": full["macro_f1_all_checkpoint_classes"],
                      "half_validation_accuracy": half["accuracy"],
                      "elapsed_seconds": time.perf_counter()-started,
                      "checkpoint": str(checkpoint), "validation_results": str(evaluation),
                      "training_directory": str(out)}
            results.append(result)
            write_json(suite/"validation_comparison.json", results)
            publish(completed_runs=[row["name"] for row in results])
            print(json.dumps(result), flush=True)
        publish(status="complete", stage="complete", child_pid=None, finished_at_utc=utc_now())
        print("Completed all training and validation runs; no test-set scoring performed.", flush=True)
    except BaseException as error:
        publish(status="failed", error=str(error), child_pid=None)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue a failed queue, preserving completed runs")
    args = parser.parse_args()
    if args.execute:
        execute(args.suite_dir, resume=args.resume)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
