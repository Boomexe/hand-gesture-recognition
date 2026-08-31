# Live prediction pipeline: implemented variable-length design

## Contract

Training, uploaded-video prediction, and live-webcam prediction share the same
feature builder and use the gesture's exact detected-frame sequence. No-hands
browser events are segmentation signals and never become model rows.

A raw frame contains 255 values: sequential debug index, right/left presence
flags, then deterministic right-hand and left-hand landmark slots. Velocity
augmentation produces 510 model values per frame.

## Runtime flow

```text
Browser frame
    -> MediaPipe landmarks + source timestamp
    -> shared GestureFeatureBuilder
    -> append one detected frame to the session gesture
    -> exact-length velocity augmentation
    -> PyTorch prediction

No-hand packet
    -> increment silence counter only
    -> finalize and reset after the configured silence threshold
```

The live state is isolated per Socket.IO session. Starting or stopping the
webcam resets both the frame buffer and motion history.

## Variable-length training

Cleaned JSON stores each gesture without padding or truncation. `GestureDataset`
keeps those separate arrays. `collate_gesture_batch` pads only to the longest
sequence in the current mini-batch because PyTorch tensors in one batch must be
rectangular.

The presence flags form the valid-frame mask. Packed LSTMs skip temporary
padding; masked batch normalization, attention, and average pooling exclude it.
The padding disappears after that mini-batch and is not stored in the dataset or
used by live inference.

## Debug artifacts

- `live_gesture_debug.csv`: exact raw detected frames.
- `live_model_input_debug.csv`: exact velocity-augmented frames sent to PyTorch.
- `live_cache_debug.csv`: compatibility copy of the same exact model input.
- `uploaded_raw_debug.csv`: exact raw uploaded-video detections.
- `uploaded_video_debug.csv`: exact uploaded-video model input.

The files have one row per detected frame. Empty/silence events do not add rows,
so there are no artificial negative indexes or padding tails.

## Configuration

`website/parameters.json` records `variable_length: true`. `max_live_frames` is
`null`; it may be set to a positive integer as an operational memory safety cap,
but it is not a trained tensor length.

## Validation

Regression tests cover:

- exact-length preprocessing;
- first-frame velocity behavior;
- deterministic hand ordering;
- variable-length collation and zero batch padding;
- model invariance to ignored trailing padding;
- checkpoint round-tripping.

Before deploying, regenerate the old fixed-length cleaned JSON files, retrain,
and save the checkpoint and parameters together. Detailed steps are in
`RETRAINING_V10.md` and the change rationale is in
`VARIABLE_LENGTH_MIGRATION.md`.