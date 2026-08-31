# Variable-length gesture migration

## Why this changed

The old pipeline forced every gesture to one global length. Short gestures gained
artificial empty rows and long gestures lost their tail. That also made live
debug files appear to contain model data that was never detected by the camera.
The PyTorch architecture does not need a fixed sequence dimension, so the global
length has been removed.

## Code changes

### Shared preprocessing

`gesture_preprocessing.raw_frame_array` validates and returns the exact detected
sequence. `build_model_input` adds velocity features without padding or
truncation. Upload and live prediction both call this path.

### Dataset and model

`gesture_model.GestureDataset` stores one two-dimensional tensor per gesture.
`collate_gesture_batch` uses `pad_sequence` to make only the current mini-batch
rectangular. Presence flags identify real rows. Packed LSTMs and the model's
masked normalization, attention, and pooling exclude padded rows.

This padding is a transport detail required to batch unequal tensors; it is not
a gesture length, is not saved into cleaned data, and is not used for single
live predictions.

### Notebook

`detect_gestures.ipynb` now:

- writes cleaned gestures directly as a list of exact-length arrays;
- loads split features as lists rather than one rectangular NumPy array;
- applies velocity augmentation gesture by gesture;
- uses `GestureDataset` and `collate_gesture_batch`;
- derives feature width from one sample, not a sequence-length dimension;
- saves `variable_length: true` and `max_live_frames: null` parameters;
- reports minimum and maximum gesture lengths instead of a model frame contract.

Stale notebook outputs were cleared so old fixed-length messages are not mistaken
for current behavior.

### Flask and browser paths

`website/app.py` sends exact-length upload and live tensors to PyTorch. Live state
stores detected frames only. `max_live_frames` is optional and disabled by
default. `index.js` already had no frame-count assumption, so no JavaScript
behavioral change was required.

### Debug CSVs

All live debug CSV rows now correspond to detected gesture frames. Silence does
not create a model row, and no padding tail is written. Batch padding exists only
inside a training `DataLoader` batch.

## What you must do

1. Activate the CUDA-enabled virtual environment.
2. Open `detect_gestures.ipynb` from the repository root.
3. If your raw split CSVs already use schema v2, skip the expensive video feature
   generation cell. Otherwise regenerate them.
4. Run the cleaning cell to replace the old fixed-length cleaned JSON files.
5. Run the remaining load, augmentation, training, evaluation, and save cells.
6. Deploy only after validation/test results are acceptable.
7. Restart `website/app.py` so it loads the newly saved checkpoint and parameters.

Retraining is required to learn from the newly retained long-sequence tails and
to keep the deployed model paired with its preprocessing contract.

## Archived TensorFlow notebook

`detect_gestures_tensorflow.ipynb` is a frozen historical backup and is not an
active training or runtime path. It retains the old TensorFlow experiment,
including its historical fixed-shape preprocessing. Do not use it to produce the
PyTorch deployment.