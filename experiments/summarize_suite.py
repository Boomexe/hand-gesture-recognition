"""Aggregate completed validation ablations without loading or scoring test data."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

NAMES = {
    "legacy_full": "Legacy features",
    "positions_full": "Wrist-centered positions",
    "scaled-positions_full": "Scale-normalized positions",
    "positions-diff_full": "Positions + frame differences",
    "positions-motion_full": "Positions + stored motion",
    "positions_prefix": "Positions + prefix training",
}


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "by_seed": {str(seed): value for seed, value in zip((17, 42, 73), values)}}


def aggregate(suite):
    status = json.loads((suite/"status.json").read_text())
    records = json.loads((suite/"validation_comparison.json").read_text())
    assert status["status"] == "complete" and len(records) == 18
    groups = {}
    input_hashes = {}
    train_ids, val_ids, feature_hashes = set(), set(), set()
    for record in records:
        key = record["variant"]+("_prefix" if record["prefix_probability"] else "_full")
        validation = suite/"validation"/Path(record["validation_results"]).name
        run_dir = suite/"runs"/Path(record.get("training_directory", record["name"])).name
        settings = json.loads((run_dir/"settings.json").read_text())
        report = json.loads((validation/"summary.json").read_text())
        train_ids.add(settings["train_ids_sha256"])
        val_ids.add(settings["validation_ids_sha256"])
        feature_hashes.add(settings["source_hashes"]["experiments/features.py"])
        assert settings["train_audit"]["evaluated_samples"] == 770
        assert settings["validation_audit"]["evaluated_samples"] == 189
        assert settings["limit_per_class"] is None
        assert all(row["count"] == 189 for row in report)
        full = next(row for row in report if row["fraction"] == 1)
        assert abs(full["accuracy"]-record["full_validation_accuracy"]) < 1e-12
        groups.setdefault(key, {})[record["seed"]] = {
            "record": record, "settings": settings,
            "fractions": {row["fraction"]: row for row in report if row["fraction"] is not None}}
        for source in (validation/"summary.json", run_dir/"settings.json"):
            input_hashes[source.relative_to(suite).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(train_ids) == len(val_ids) == len(feature_hashes) == 1
    assert set(groups) == set(NAMES)
    summary = {}
    for key in NAMES:
        runs = groups[key]
        assert set(runs) == {17, 42, 73}
        parameters = {run["settings"]["parameter_count"] for run in runs.values()}
        assert len(parameters) == 1
        metrics = {metric: stats([runs[seed]["fractions"][1][metric] for seed in (17, 42, 73)])
                   for metric in ("accuracy", "macro_f1_all_checkpoint_classes", "top5_accuracy",
                                  "existing_threshold_0.8_coverage", "existing_threshold_0.8_accepted_accuracy")}
        prefixes = {str(q): stats([runs[seed]["fractions"][q]["accuracy"] for seed in (17, 42, 73)])
                    for q in sorted(runs[17]["fractions"])}
        summary[key] = {"label": NAMES[key], "parameter_count": parameters.pop(),
                        "full_sequence": metrics, "prefix_accuracy": prefixes,
                        "best_epochs": {str(seed): runs[seed]["record"]["best_epoch"] for seed in (17, 42, 73)}}
    effects = {}
    for q in summary["positions_full"]["prefix_accuracy"]:
        effects[q] = stats([groups["positions_prefix"][seed]["fractions"][float(q)]["accuracy"]
                            - groups["positions_full"][seed]["fractions"][float(q)]["accuracy"]
                            for seed in (17, 42, 73)])
    return {"split": "validation", "classes": 52, "train_clips": 770, "validation_clips": 189,
            "seeds": [17, 42, 73], "configurations": summary, "paired_prefix_training_effect": effects,
            "input_sha256": input_hashes, "shared_training_ids_sha256": train_ids.pop(),
            "shared_validation_ids_sha256": val_ids.pop(), "shared_features_source_sha256": feature_hashes.pop(),
            "interpretation": "Sample SD is across three training seeds, not a confidence interval across signers. Models were selected on this validation set; results are exploratory and not held-out test or live-webcam estimates."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = aggregate(args.suite)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"ablation_summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = ["# Completed validation ablations", "",
             "52 classes; 770 training clips; 189 validation clips; seeds 17, 42, 73. Means +/- sample SD across training seeds. No new test-set evaluation.", "",
             "| Configuration | Parameters | Full accuracy (%) | Full macro F1 | Half-prefix accuracy (%) |",
             "|---|---:|---:|---:|---:|"]
    for value in data["configurations"].values():
        full = value["full_sequence"]["accuracy"]
        f1 = value["full_sequence"]["macro_f1_all_checkpoint_classes"]
        half = value["prefix_accuracy"]["0.5"]
        lines.append(f"| {value['label']} | {value['parameter_count']:,} | {100*full['mean']:.2f} +/- {100*full['sample_sd']:.2f} | {f1['mean']:.4f} +/- {f1['sample_sd']:.4f} | {100*half['mean']:.2f} +/- {100*half['sample_sd']:.2f} |")
    lines += ["", "## Matched prefix-training comparison", "",
              "| Detected frames observed | Complete-only accuracy (%) | Prefix-trained accuracy (%) | Paired mean gain (percentage points) |",
              "|---|---:|---:|---:|"]
    for q, effect in data["paired_prefix_training_effect"].items():
        baseline = data["configurations"]["positions_full"]["prefix_accuracy"][q]
        augmented = data["configurations"]["positions_prefix"]["prefix_accuracy"][q]
        lines.append(f"| {100*float(q):.0f}% | {100*baseline['mean']:.2f} +/- {100*baseline['sample_sd']:.2f} | {100*augmented['mean']:.2f} +/- {100*augmented['sample_sd']:.2f} | {100*effect['mean']:+.2f} |")
    lines += ["", "## Individual full-sequence validation accuracies", "",
              "| Configuration | Seed 17 (%) | Seed 42 (%) | Seed 73 (%) |", "|---|---:|---:|---:|"]
    for value in data["configurations"].values():
        by_seed = value["full_sequence"]["accuracy"]["by_seed"]
        lines.append(f"| {value['label']} | {100*by_seed['17']:.2f} | {100*by_seed['42']:.2f} | {100*by_seed['73']:.2f} |")
    lines += ["", data["interpretation"], ""]
    text = "\n".join(lines)
    (args.output/"ABLATION_RESULTS.md").write_text(text, encoding="utf-8")
    print(text)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, color, label in (("positions_full", "#1769aa", "Complete-only training"),
                               ("positions_prefix", "#d65f00", "Prefix-augmented training")):
        curve = data["configurations"][key]["prefix_accuracy"]
        x = np.array([float(q)*100 for q in curve])
        y = np.array([v["mean"]*100 for v in curve.values()])
        sd = np.array([v["sample_sd"]*100 for v in curve.values()])
        ax.plot(x, y, marker="o", color=color, label=label)
        ax.fill_between(x, np.maximum(0, y-sd), np.minimum(100, y+sd), color=color, alpha=0.15)
    ax.set(xlabel="Observed detected frames (%)", ylabel="Validation accuracy (%)", ylim=(0, 100),
           title="Partial-gesture recognition: matched position-only models")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.text(0.5, 0.01, "52 classes; 189 validation clips; mean +/- sample SD across 3 training seeds", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.output/"prefix_training_comparison.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
