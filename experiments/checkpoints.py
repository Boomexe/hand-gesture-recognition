"""Immutable training checkpoints with a small replaceable best-file pointer."""

import json
from pathlib import Path

from experiments.json_io import atomic_write_json

from gesture_model import save_gesture_checkpoint


def write_pointer(path: Path, payload) -> None:
    # Checkpoints are never overwritten. Only this small JSON pointer is replaced.
    atomic_write_json(path, payload)


def save_best_checkpoint(directory: Path, epoch: int, model, metadata: dict) -> Path:
    directory = Path(directory)
    checkpoint_dir = directory/"checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir/f"epoch_{epoch:04d}.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite an immutable checkpoint: {checkpoint}")
    save_gesture_checkpoint(checkpoint, model, metadata)
    write_pointer(directory/"best_checkpoint.json",
                  {"checkpoint": checkpoint.relative_to(directory).as_posix(), "epoch": epoch,
                   "validation_loss": metadata.get("validation_loss"),
                   "validation_accuracy": metadata.get("validation_accuracy")})
    return checkpoint


def resolve_best_checkpoint(directory: Path) -> Path:
    directory = Path(directory).resolve()
    pointer = directory/"best_checkpoint.json"
    if pointer.exists():
        value = json.loads(pointer.read_text(encoding="utf-8"))
        checkpoint = (directory/value["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(directory):
            raise ValueError("Best-checkpoint pointer escapes the run directory")
    else:
        checkpoint = directory/"best.pt"  # Existing completed experiments remain readable.
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint
