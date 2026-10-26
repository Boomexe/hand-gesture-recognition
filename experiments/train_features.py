"""Train one feature or prefix-training ablation without modifying the notebook/app."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.evaluate_prefixes import class_mapping, read_samples
from experiments.features import VARIANTS, feature_view
from experiments.checkpoints import save_best_checkpoint, write_pointer
from gesture_model import (GestureClassifier, GestureModelConfig, collate_gesture_batch,
                           evaluate_model)


class FeatureDataset(Dataset):
    def __init__(self, samples, variant, prefix_probability=0.0):
        self.samples = samples
        self.variant = variant
        self.prefix_probability = prefix_probability

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        raw = sample["raw"]
        if random.random() < self.prefix_probability:
            raw = raw[:random.randint(1, len(raw))]
        return torch.from_numpy(feature_view(raw, self.variant)), torch.tensor(sample["label"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="legacy")
    parser.add_argument("--prefix-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new run directory")
    if not 0 <= args.prefix_probability <= 1 or min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("Invalid training settings")
    if args.learning_rate <= 0 or (args.limit_per_class is not None and args.limit_per_class < 1):
        parser.error("Learning rate and optional sample limit must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    parameters = json.loads(args.parameters.read_text())
    mapping = class_mapping({}, parameters, len(parameters["numbered_unique_words"]))
    print("Loading the frozen vocabulary's training and validation sequences...", flush=True)
    train_samples, train_audit = read_samples(args.data_dir/"cleaned_train_data.h5", mapping, args.limit_per_class, args.seed)
    val_samples, val_audit = read_samples(args.data_dir/"cleaned_val_data.h5", mapping, args.limit_per_class, args.seed)
    if any(n == 0 for n in train_audit["support_in_file"].values()):
        raise ValueError("Every configured class must have training examples")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name if args.device == "auto" else args.device)
    train_set = FeatureDataset(train_samples, args.variant, args.prefix_probability)
    val_set = FeatureDataset(val_samples, args.variant)
    loader_options = dict(batch_size=args.batch_size, num_workers=0,
                          pin_memory=device.type == "cuda", collate_fn=collate_gesture_batch)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=shuffle_generator, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    config = GestureModelConfig(input_size=feature_view(train_samples[0]["raw"], args.variant).shape[1],
                                num_classes=len(mapping))
    model = GestureClassifier(config).to(device)
    print(f"Device={device}; train={len(train_set)}; validation={len(val_set)}; "
          f"variant={args.variant}; prefix_probability={args.prefix_probability}; seed={args.seed}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    args.output.mkdir(parents=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["feature_variant"] = args.variant
    settings["train_audit"] = train_audit
    settings["validation_audit"] = val_audit
    settings["parameter_count"] = sum(p.numel() for p in model.parameters())
    settings["train_ids_sha256"] = hashlib.sha256(json.dumps([s['id'] for s in train_samples]).encode()).hexdigest()
    settings["validation_ids_sha256"] = hashlib.sha256(json.dumps([s['id'] for s in val_samples]).encode()).hexdigest()
    settings["source_hashes"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ("gesture_model.py", "experiments/features.py", "experiments/train_features.py", "experiments/checkpoints.py")}
    settings["torch"] = str(torch.__version__)
    settings["notes"] = ["Validation selection uses complete sequences only for both training policies.",
                         "Seeds are recorded; exact GPU determinism is not guaranteed.",
                         "HDF5 has no signer identifiers; signer separation must be verified from source manifests."]
    (args.output/"settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    history = []
    best_loss, stale = float("inf"), 0
    best_state = None
    best_checkpoint = None
    for epoch in range(1, args.epochs+1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()*len(labels)
            correct += logits.argmax(1).eq(labels).sum().item()
            count += len(labels)
        val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss/count, "train_accuracy": correct/count,
               "validation_loss": val_loss, "validation_accuracy": val_accuracy,
               "elapsed_seconds": time.perf_counter()-epoch_started}
        history.append(row)
        write_pointer(args.output/"history.json", history)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            best_checkpoint = save_best_checkpoint(args.output, epoch, model,
                                      {"parameters": {**parameters, "data_per_frame": config.input_size}, "experiment": settings,
                                       "epoch": epoch, "validation_loss": val_loss,
                                       "validation_accuracy": val_accuracy})
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_checkpoint is None:
        raise RuntimeError("Training finished without a valid checkpoint")
    (args.output/"training_complete.json").write_text(
        json.dumps({"epochs_run": len(history), "checkpoint": str(best_checkpoint)}, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint to {best_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
