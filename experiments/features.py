"""Causal feature views for controlled ablations of existing 255-channel data."""

import numpy as np

from gesture_preprocessing import add_velocity


VARIANTS = ("legacy", "positions", "scaled-positions", "positions-diff", "positions-motion")


def feature_view(raw: np.ndarray, variant: str = "legacy") -> np.ndarray:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    if variant == "legacy":
        return add_velocity(raw[None, ...])[0].copy()
    hands = raw[:, 3:].reshape(len(raw), 2, 21, 6)
    positions = hands[..., :3].copy()
    present = raw[:, 1:3].astype(bool)
    if variant == "scaled-positions":
        # Landmark 9 is the middle-finger MCP, relative to wrist landmark 0.
        scale = np.linalg.norm(positions[:, :, 9], axis=-1)
        positions /= np.maximum(scale, 1e-6)[..., None, None]
    positions *= present[..., None, None]
    if variant == "positions-diff":
        motion = np.diff(positions, axis=0, prepend=positions[:1])
        continuous = present.copy()
        continuous[0] = False
        continuous[1:] &= present[:-1]
        motion *= continuous[..., None, None]
    elif variant == "positions-motion":
        # Retain stored per-second values; their source provenance must be audited.
        motion = hands[..., 3:].copy()
        continuous = present.copy()
        continuous[0] = False
        continuous[1:] &= present[:-1]
        motion *= continuous[..., None, None]
    else:
        motion = None
    # Keep presence at columns 1 and 2 for the existing model mask.
    header = np.zeros((len(raw), 3), dtype=np.float32)
    header[:, 1:3] = raw[:, 1:3]
    blocks = [header, positions.reshape(len(raw), -1)]
    if motion is not None:
        blocks.append(motion.reshape(len(raw), -1))
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
