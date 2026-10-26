"""Evaluate gesture checkpoints on held-out prefixes of raw HDF5 data.

Run from the repository root. This preserves the checkpoint's feature schema;
feature removal/transformation requires separate training. Ablation checkpoints
record their feature view; existing checkpoints retain the legacy view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys

import h5py
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gesture_model import load_gesture_checkpoint
from gesture_preprocessing import RAW_FEATURES_PER_FRAME
from experiments.features import feature_view
from experiments.checkpoints import resolve_best_checkpoint


def class_mapping(metadata: dict, external: dict | None, num_classes: int) -> dict:
    embedded = metadata.get("parameters", {}).get("numbered_unique_words")
    supplied = external.get("numbered_unique_words") if external else None
    if embedded is not None and supplied is not None and embedded != supplied:
        raise ValueError("External class map differs from the checkpoint's ordered class map")
    mapping = embedded if embedded is not None else supplied
    if mapping is None:
        raise ValueError("Checkpoint has no class map; provide its original --parameters file")
    mapping = {str(word): int(index) for word, index in mapping.items()}
    if len(mapping) != num_classes or sorted(mapping.values()) != list(range(num_classes)):
        raise ValueError("Class map must cover exactly every checkpoint output once")
    return mapping


def validate_raw(raw: np.ndarray) -> None:
    if raw.ndim != 2 or raw.shape[1] != RAW_FEATURES_PER_FRAME or len(raw) == 0:
        raise ValueError(f"Expected nonempty (T, {RAW_FEATURES_PER_FRAME}), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Raw features contain nonfinite values")
    flags = raw[:, 1:3]
    if not np.isin(flags, (0, 1)).all():
        raise ValueError("Hand presence flags must be binary")
    if not flags.any(axis=1).all():
        raise ValueError("Raw sample contains all-absent frames; regenerate exact-length data")


def prefix_input(raw: np.ndarray, *, fraction: float | None = None,
                 frames: int | None = None, variant: str = "legacy") -> torch.Tensor:
    validate_raw(raw)
    if (fraction is None) == (frames is None):
        raise ValueError("Supply exactly one prefix length specification")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("Fraction must be in (0, 1]")
    if frames is not None and frames < 1:
        raise ValueError("Frames must be positive")
    count = math.ceil(len(raw) * fraction) if fraction is not None else frames
    # Truncate before feature augmentation: no future observations are used.
    observed = raw[:min(len(raw), count)]
    return torch.from_numpy(feature_view(observed, variant))


def read_samples(path: Path, mapping: dict, limit_per_class: int | None, seed: int):
    buckets = {word: [] for word in mapping}
    with h5py.File(path, "r") as h5:
        total = int(h5.attrs["num_samples"])
        for index in range(total):
            group = h5[str(index)]
            word = group.attrs["word"]
            if isinstance(word, bytes):
                word = word.decode("utf-8")
            if word in buckets:
                buckets[word].append(index)
        rng = random.Random(seed)
        selected = []
        for word in sorted(buckets, key=mapping.get):
            candidates = buckets[word]
            if limit_per_class is not None and len(candidates) > limit_per_class:
                candidates = sorted(rng.sample(candidates, limit_per_class))
            selected.extend(candidates)
        samples = []
        for index in sorted(selected):
            group = h5[str(index)]
            word = group.attrs["word"]
            if isinstance(word, bytes):
                word = word.decode("utf-8")
            raw = group["data"][:].astype(np.float32, copy=False)
            try:
                validate_raw(raw)
            except ValueError as error:
                raise ValueError(f"Invalid HDF5 sample {index}: {error}") from error
            samples.append({"id": str(index), "clip_name": str(group.attrs.get("clip_name", "")),
                            "word": word, "label": mapping[word], "raw": raw})
    if not samples:
        raise ValueError("No held-out gestures match checkpoint vocabulary")
    return samples, {"total_hdf5_samples": total,
                     "vocabulary_matched_samples": sum(map(len, buckets.values())),
                     "evaluated_samples": len(samples),
                     "support_in_file": {word: len(indices) for word, indices in buckets.items()}}


def wilson_interval(correct: int, count: int):
    z = 1.959963984540054
    p = correct / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [center-half, center+half]


def evaluate(model, samples, device, names, batch_size, fraction=None, frames=None, variant="legacy"):
    labels, predictions, confidence, top5, lengths = [], [], [], [], []
    with torch.inference_mode():
        for offset in range(0, len(samples), batch_size):
            batch = samples[offset:offset+batch_size]
            features = [prefix_input(s["raw"], fraction=fraction, frames=frames, variant=variant) for s in batch]
            values = pad_sequence(features, batch_first=True).to(device)
            probabilities = model(values).softmax(dim=1)
            predicted = probabilities.argmax(dim=1)
            y = torch.tensor([s["label"] for s in batch], device=device)
            top = probabilities.topk(min(5, len(names)), dim=1).indices
            labels.extend(y.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            confidence.extend(probabilities.max(dim=1).values.cpu().tolist())
            top5.extend(top.eq(y[:, None]).any(dim=1).cpu().tolist())
            lengths.extend(len(x) for x in features)
    report = classification_report(labels, predictions, labels=list(range(len(names))),
                                   target_names=names, zero_division=0, output_dict=True)
    correct = np.equal(labels, predictions)
    present = [name for name in names if report[name]["support"] > 0]
    accepted = np.asarray(confidence) >= 0.8
    metrics = {"fraction": fraction, "frame_cap": frames, "count": len(labels),
               "accuracy": float(correct.mean()), "top5_accuracy": float(np.mean(top5)),
               "macro_f1_all_checkpoint_classes": report["macro avg"]["f1-score"],
               "macro_f1_supported_classes": float(np.mean([report[n]["f1-score"] for n in present])),
               "weighted_f1": report["weighted avg"]["f1-score"],
               "accuracy_wilson95_independent_clip_assumption": wilson_interval(int(correct.sum()), len(labels)),
               "existing_threshold_0.8_coverage": float(accepted.mean()),
               "existing_threshold_0.8_accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
               "mean_observed_frames": float(np.mean(lengths)),
               "clips_shorter_than_frame_cap": sum(len(s["raw"]) < frames for s in samples) if frames else None,
               "per_class": report,
               "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(names)))).tolist()}
    records = [{"sample_id": s["id"], "clip_name": s["clip_name"], "true_label": y,
                "predicted_label": pred, "confidence": conf, "observed_frames": count,
                "total_detected_frames": len(s["raw"])}
               for s, y, pred, conf, count in zip(samples, labels, predictions, confidence, lengths)]
    return metrics, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fractions", type=float, nargs="+", default=[i/10 for i in range(1, 11)])
    parser.add_argument("--frame-counts", type=int, nargs="*", default=[8, 13, 20, 31])
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.checkpoint.is_dir():
        args.checkpoint = resolve_best_checkpoint(args.checkpoint)
    if args.batch_size < 1 or (args.limit_per_class is not None and args.limit_per_class < 1):
        parser.error("Batch size and optional limit must be positive")
    if any(not 0 < q <= 1 for q in args.fractions) or any(n < 1 for n in args.frame_counts):
        parser.error("Fractions must be in (0, 1] and frame counts positive")
    if 1.0 not in args.fractions:
        args.fractions.append(1.0)
    if args.output.exists():
        parser.error("Output already exists; use a new directory to preserve earlier results")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, metadata = load_gesture_checkpoint(args.checkpoint, device)
    external = json.loads(args.parameters.read_text()) if args.parameters else None
    mapping = class_mapping(metadata, external, model.config.num_classes)
    variant = metadata.get("experiment", {}).get("feature_variant", "legacy")
    names = [word for word, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    samples, audit = read_samples(args.data, mapping, args.limit_per_class, args.seed)
    if feature_view(samples[0]["raw"], variant).shape[1] != model.config.input_size:
        raise ValueError("Checkpoint input dimension differs from its recorded feature representation")
    print(f"Device: {device}; classes: {len(names)}; audit: {audit['evaluated_samples']} / "
          f"{audit['total_hdf5_samples']} HDF5 gestures", flush=True)
    args.output.mkdir(parents=True)
    manifest = {"checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "data": str(args.data.resolve()), "data_bytes": args.data.stat().st_size,
                "data_mtime_ns": args.data.stat().st_mtime_ns,
                "selected_ids_sha256": hashlib.sha256(json.dumps([s['id'] for s in samples]).encode()).hexdigest(),
                "class_mapping": mapping, "checkpoint_metadata": metadata, "audit": audit,
                "python": platform.python_version(), "torch": torch.__version__,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "model_source_sha256": hashlib.sha256((ROOT/'gesture_model.py').read_bytes()).hexdigest(),
                "preprocessing_source_sha256": hashlib.sha256((ROOT/'gesture_preprocessing.py').read_bytes()).hexdigest(),
                "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "seed": args.seed, "batch_size": args.batch_size, "limit_per_class": args.limit_per_class,
                "feature_variant": variant,
                "feature_source_sha256": hashlib.sha256((ROOT/'experiments/features.py').read_bytes()).hexdigest(),
                "notes": ["Only labels in the checkpoint vocabulary are evaluated; all matching clips are included unless limited.",
                          "Completion fraction counts detected frames, not wall-clock duration.",
                          "Signer identities and timestamp provenance are unavailable in this HDF5 format.",
                          "Wilson intervals assume independent clips; do not treat them as signer-clustered intervals.",
                          "Threshold 0.8 is the existing app setting, not selected or calibrated on this test set.",
                          "No inference latency is claimed from batched evaluation.",
                          "Stored position/motion values retain their original preprocessing; this does not validate their provenance."]}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries = []
    full_records = None
    with (args.output/"predictions.jsonl").open("w", encoding="utf-8") as stream:
        conditions = [(q, None) for q in sorted(set(args.fractions))]
        conditions += [(None, n) for n in sorted(set(args.frame_counts))]
        for fraction, frames in conditions:
            metrics, records = evaluate(model, samples, device, names, args.batch_size, fraction, frames, variant)
            label = f"fraction_{fraction:.2f}" if fraction is not None else f"first_{frames}_frames"
            (args.output/f"{label}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            for record in records:
                stream.write(json.dumps({"condition": label, **record}) + "\n")
            if fraction == 1:
                full_records = records
            summaries.append({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")})
            print(f"{label}: accuracy={metrics['accuracy']:.3%}; "
                  f"macro F1={metrics['macro_f1_all_checkpoint_classes']:.4f}", flush=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    full_by_id = {r["sample_id"]: r for r in full_records}
    # Paired correctness transitions are more useful than comparing two unrelated accuracies.
    paired = []
    for fraction, frames in conditions:
        label = f"fraction_{fraction:.2f}" if fraction is not None else f"first_{frames}_frames"
        transitions = {"both_correct": 0, "prefix_wrong_full_correct": 0,
                       "prefix_correct_full_wrong": 0, "both_wrong": 0}
        with (args.output/"predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                if r["condition"] != label:
                    continue
                full = full_by_id[r["sample_id"]]
                early_correct = r["predicted_label"] == r["true_label"]
                full_correct = full["predicted_label"] == full["true_label"]
                key = ("both_correct" if early_correct and full_correct else
                       "prefix_wrong_full_correct" if full_correct else
                       "prefix_correct_full_wrong" if early_correct else "both_wrong")
                transitions[key] += 1
        paired.append({"condition": label, **transitions})
    (args.output/"paired_correctness.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; numerical results saved without a plot", flush=True)
    else:
        points = [s for s in summaries if s["fraction"] is not None]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([p["fraction"]*100 for p in points], [p["accuracy"]*100 for p in points], "o-", label="Top-1 accuracy")
        ax.plot([p["fraction"]*100 for p in points], [p["macro_f1_all_checkpoint_classes"]*100 for p in points], "s--", label="Macro F1")
        ax.set(xlabel="Detected-frame completion (%)", ylabel="Metric (%)", ylim=(0, 100),
               title=f"Held-out prefix recognition ({len(names)} classes, n={len(samples)})")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output/"prefix_accuracy.png", dpi=200)
        plt.close(fig)
    print(f"Results saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
