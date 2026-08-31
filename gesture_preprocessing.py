"""Shared gesture feature construction for training, uploads, and live input."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


LANDMARK_COUNT = 21
VALUES_PER_LANDMARK = 6
HAND_FEATURES = LANDMARK_COUNT * VALUES_PER_LANDMARK
RAW_FEATURES_PER_FRAME = 3 + (2 * HAND_FEATURES)


def add_velocity(values: np.ndarray) -> np.ndarray:
    """Append frame-to-frame differences to a batch of gesture sequences."""
    velocity = np.diff(values, axis=1, prepend=values[:, :1, :])
    return np.concatenate([values, velocity], axis=-1)


def handedness_to_index(handedness: Any) -> int | None:
    """Normalize MediaPipe Python or JavaScript handedness to 0=right, 1=left."""
    if isinstance(handedness, (list, tuple)):
        if not handedness:
            return None
        handedness = handedness[0]

    if isinstance(handedness, dict):
        label = handedness.get("categoryName") or handedness.get("displayName")
        index = handedness.get("index")
    else:
        label = getattr(handedness, "category_name", None) or getattr(
            handedness, "display_name", None
        )
        index = getattr(handedness, "index", None)

    if isinstance(label, str):
        normalized = label.casefold()
        if normalized == "right":
            return 0
        if normalized == "left":
            return 1

    if index in (0, 1):
        return int(index)
    return None


def _xyz(landmark: Any) -> tuple[float, float, float]:
    if isinstance(landmark, dict):
        return (
            float(landmark["x"]),
            float(landmark["y"]),
            float(landmark["z"]),
        )
    return float(landmark.x), float(landmark.y), float(landmark.z)


@dataclass
class GestureFeatureBuilder:
    """Stateful builder for the legacy 255-value raw frame schema.

    Landmark positions are wrist-relative. The wrist's position features remain
    zero, while its motion uses the absolute wrist position. Other landmark
    motion uses wrist-relative positions consistently. Motion is expressed per
    second so video and webcam frame rates share the same feature scale.
    """

    last_positions: dict[int, list[tuple[float, float, float]] | None] = field(
        default_factory=lambda: {0: None, 1: None}
    )
    last_timestamps_ms: dict[int, float | None] = field(
        default_factory=lambda: {0: None, 1: None}
    )

    def reset(self) -> None:
        self.last_positions = {0: None, 1: None}
        self.last_timestamps_ms = {0: None, 1: None}

    def _hand_features(
        self,
        landmarks: Sequence[Any],
        hand_index: int,
        timestamp_ms: float | None,
    ) -> list[float]:
        if len(landmarks) != LANDMARK_COUNT:
            raise ValueError(
                f"Expected {LANDMARK_COUNT} landmarks, got {len(landmarks)}"
            )

        absolute_positions = [_xyz(landmark) for landmark in landmarks]
        origin_x, origin_y, origin_z = absolute_positions[0]

        comparison_positions = [absolute_positions[0]]
        comparison_positions.extend(
            (x - origin_x, y - origin_y, z - origin_z)
            for x, y, z in absolute_positions[1:]
        )

        previous = self.last_positions[hand_index]
        previous_timestamp_ms = self.last_timestamps_ms[hand_index]
        if previous is None:
            previous = comparison_positions

        if timestamp_ms is None or previous_timestamp_ms is None:
            elapsed_seconds = None
        else:
            elapsed_seconds = (timestamp_ms - previous_timestamp_ms) / 1000.0
            if elapsed_seconds <= 0:
                elapsed_seconds = None

        features: list[float] = []
        for landmark_index, (x, y, z) in enumerate(absolute_positions):
            rel_x = x - origin_x
            rel_y = y - origin_y
            rel_z = z - origin_z

            current_for_motion = comparison_positions[landmark_index]
            previous_for_motion = previous[landmark_index]
            if elapsed_seconds is None:
                speed_x = speed_y = speed_z = 0.0
            else:
                speed_x = (
                    current_for_motion[0] - previous_for_motion[0]
                ) / elapsed_seconds
                speed_y = (
                    current_for_motion[1] - previous_for_motion[1]
                ) / elapsed_seconds
                speed_z = (
                    current_for_motion[2] - previous_for_motion[2]
                ) / elapsed_seconds

            features.extend(
                [rel_x, rel_y, rel_z, speed_x, speed_y, speed_z]
            )

        self.last_positions[hand_index] = comparison_positions
        self.last_timestamps_ms[hand_index] = timestamp_ms
        return features

    def build_frame(
        self,
        hands: Iterable[tuple[Any, Sequence[Any]]],
        frame_index: int,
        timestamp_ms: float | None = None,
    ) -> list[float]:
        """Build one frame with deterministic right-hand then left-hand slots."""
        features_by_hand: dict[int, list[float]] = {}

        for handedness, landmarks in hands:
            hand_index = handedness_to_index(handedness)
            if hand_index is None:
                continue
            features_by_hand[hand_index] = self._hand_features(
                landmarks, hand_index, timestamp_ms
            )

        right_present = int(0 in features_by_hand)
        left_present = int(1 in features_by_hand)
        row: list[float] = [float(frame_index), right_present, left_present]
        row.extend(features_by_hand.get(0, [0.0] * HAND_FEATURES))
        row.extend(features_by_hand.get(1, [0.0] * HAND_FEATURES))

        if len(row) != RAW_FEATURES_PER_FRAME:
            raise ValueError(
                f"Raw frame has {len(row)} values; expected "
                f"{RAW_FEATURES_PER_FRAME}"
            )
        return row


def raw_frame_array(frames: Sequence[Sequence[float]]) -> np.ndarray:
    """Return an exact-length raw gesture array without padding or truncation."""
    if not frames:
        raise ValueError("At least one detected frame is required")

    raw = np.asarray(frames, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != RAW_FEATURES_PER_FRAME:
        raise ValueError(
            f"Expected gesture shape (frames, {RAW_FEATURES_PER_FRAME}), "
            f"got {raw.shape}"
        )
    return raw


def build_model_input(
    frames: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact-length raw frames and velocity-augmented model input."""
    raw = raw_frame_array(frames)
    model_input = add_velocity(raw[np.newaxis, ...])
    return raw, model_input