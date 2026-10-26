"""PyTorch gesture classifier, training helpers, and checkpoint utilities."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GestureModelConfig:
    input_size: int
    num_classes: int
    lstm1_hidden_size: int = 128
    lstm2_hidden_size: int = 64
    attention_heads: int = 4
    feed_forward_size: int = 128
    classifier_hidden_size: int = 64
    lstm_dropout: float = 0.3
    attention_dropout: float = 0.1
    classifier_dropout: float = 0.2


def valid_frame_mask(inputs: torch.Tensor) -> torch.Tensor:
    """Return the trailing-padding mask from raw presence flags.

    The augmented schema keeps original raw values in the first half, so
    right/left presence remain at columns 1 and 2.
    """
    if inputs.ndim != 3 or inputs.shape[-1] < 3:
        raise ValueError(
            "Expected inputs shaped (batch, frames, features) with presence flags"
        )
    mask = inputs[:, :, 1].ne(0) | inputs[:, :, 2].ne(0)
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("Every gesture must contain at least one detected frame")
    return mask


class MaskedBatchNorm1d(nn.Module):
    """Batch-normalize feature vectors without including padded timesteps."""

    def __init__(self, features: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(features)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(values)
        output[mask] = self.batch_norm(values[mask])
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        feed_forward_size: int,
        dropout: float,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, embed_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention, _ = self.attention(
            values,
            values,
            values,
            key_padding_mask=~mask,
            need_weights=False,
        )
        values = self.norm1(values + self.attention_dropout(attention))
        values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
        feed_forward = self.feed_forward(values)
        values = self.norm2(
            values + self.feed_forward_dropout(feed_forward)
        )
        return values.masked_fill(~mask.unsqueeze(-1), 0.0)


class GestureClassifier(nn.Module):
    def __init__(self, config: GestureModelConfig):
        super().__init__()
        self.config = config
        self.lstm1 = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.lstm1_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        lstm1_features = config.lstm1_hidden_size * 2
        self.batch_norm1 = MaskedBatchNorm1d(lstm1_features)
        self.dropout1 = nn.Dropout(config.lstm_dropout)

        self.lstm2 = nn.LSTM(
            input_size=lstm1_features,
            hidden_size=config.lstm2_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        lstm2_features = config.lstm2_hidden_size * 2
        self.batch_norm2 = MaskedBatchNorm1d(lstm2_features)
        self.dropout2 = nn.Dropout(config.lstm_dropout)
        self.transformer = TransformerBlock(
            embed_dim=lstm2_features,
            heads=config.attention_heads,
            feed_forward_size=config.feed_forward_size,
            dropout=config.attention_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm2_features, config.classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_size, config.num_classes),
        )

    @staticmethod
    def _packed_lstm(
        lstm: nn.LSTM,
        values: torch.Tensor,
        lengths: torch.Tensor,
        total_frames: int,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            values,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=total_frames,
        )
        return output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mask = valid_frame_mask(inputs)
        lengths = mask.sum(dim=1)
        total_frames = inputs.shape[1]

        values = self._packed_lstm(
            self.lstm1, inputs, lengths, total_frames
        )
        values = self.dropout1(self.batch_norm1(values, mask))
        values = values.masked_fill(~mask.unsqueeze(-1), 0.0)

        values = self._packed_lstm(
            self.lstm2, values, lengths, total_frames
        )
        values = self.dropout2(self.batch_norm2(values, mask))
        values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
        values = self.transformer(values, mask)

        mask_values = mask.unsqueeze(-1).to(values.dtype)
        pooled = (values * mask_values).sum(dim=1) / lengths.unsqueeze(1)
        return self.classifier(pooled)


class GestureDataset(Dataset):
    """Dataset for variable-length, velocity-augmented gesture arrays."""

    def __init__(self, features, labels):
        if len(features) != len(labels):
            raise ValueError("Feature and label counts must match")
        self.features = [
            torch.as_tensor(values, dtype=torch.float32) for values in features
        ]
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        if not self.features:
            raise ValueError("GestureDataset cannot be empty")
        feature_size = self.features[0].shape[1]
        for values in self.features:
            if values.ndim != 2 or values.shape[0] == 0:
                raise ValueError(
                    "Every gesture must have shape (detected_frames, features)"
                )
            if values.shape[1] != feature_size:
                raise ValueError("Every gesture must use the same feature schema")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def collate_gesture_batch(batch):
    """Pad only to the longest gesture in the current batch."""
    features, labels = zip(*batch)
    return (
        pad_sequence(features, batch_first=True, padding_value=0.0),
        torch.stack(labels),
    )

def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_gesture_checkpoint(
    path: str | Path,
    model: GestureClassifier,
    metadata: dict | None = None,
) -> None:
    checkpoint = {
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, Path(path))


def load_gesture_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
) -> tuple[GestureClassifier, dict]:
    target_device = torch.device(device) if device is not None else select_device()
    checkpoint = torch.load(
        Path(path),
        map_location=target_device,
        weights_only=True,
    )
    model = GestureClassifier(GestureModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()
    return model, checkpoint.get("metadata", {})


def evaluate_model(
    model: GestureClassifier,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    loss_function: nn.Module | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    criterion = loss_function or nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    predictions = []
    labels = []

    with torch.inference_mode():
        for inputs, targets in batches:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            predicted = logits.argmax(dim=1)

            batch_size = targets.shape[0]
            total_loss += float(loss) * batch_size
            total_correct += int((predicted == targets).sum())
            total_examples += batch_size
            predictions.append(predicted.cpu())
            labels.append(targets.cpu())

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        torch.cat(predictions).numpy(),
        torch.cat(labels).numpy(),
    )


def train_gesture_model(
    model: GestureClassifier,
    train_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int = 150,
    patience: int = 15,
    learning_rate: float = 0.001,
    name: str = "hand_gesture_modelv11"
) -> dict[str, list[float]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_state = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for inputs, targets in train_batches:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = targets.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum())
            total_examples += batch_size

        train_loss = total_loss / total_examples
        train_accuracy = total_correct / total_examples

        validation_loss, validation_accuracy, _, _ = evaluate_model(
            model,
            validation_batches,
            device,
            criterion,
        )

        history["loss"].append(train_loss)
        history["accuracy"].append(train_accuracy)
        history["val_loss"].append(validation_loss)
        history["val_accuracy"].append(validation_accuracy)

        print(
            f"Epoch {epoch:03d}: "
            f"loss={train_loss:.4f}, "
            f"accuracy={train_accuracy:.3f}, "
            f"val_loss={validation_loss:.4f}, "
            f"val_accuracy={validation_accuracy:.3f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            save_gesture_checkpoint(
                f"models/{name}_best.pt",
                model,
                metadata={
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                },
            )

            print(f"Saved new best checkpoint at epoch {epoch}")

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history
