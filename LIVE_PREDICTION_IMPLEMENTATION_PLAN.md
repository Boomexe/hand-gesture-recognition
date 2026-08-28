# Live Prediction Pipeline Implementation Plan

## Goal

Make training, uploaded-video, and live-webcam prediction use the same feature-generation contract so that equivalent landmark sequences produce equivalent model tensors.

This plan assumes a clean dataset regeneration and model retraining. Correcting feature generation while retaining model v9 would leave the model trained on the old preprocessing defects.

## Important padding clarification

The current notebook does not produce completely zero-filled padding rows. It stores `current_frame + 1` in the frame-index column and sets the other values to zero. Because Keras `Masking(mask_value=0.0)` only masks a row when every feature is zero, those rows are not actually masked.

For the retrained model, frame index should be removed from the learned features and retained only as debug metadata. Model padding can then be genuinely all-zero.

## 1. Define one preprocessing contract

Document exactly what one model frame contains:

- `right_present` and `left_present` flags.
- Twenty-one right-hand landmarks.
- Twenty-one left-hand landmarks.
- Fixed right-then-left slot ordering.
- Wrist-relative landmark positions.
- One consistent motion calculation.
- True all-zero model padding.
- Timestamp and sequential frame index as debug metadata, not model features.

Motion must be independent of input frame rate. Either normalize motion by elapsed time or resample every source to a fixed rate before calculating differences. Timestamp-normalized motion is the simpler initial design for variable-FPS videos and webcams.

## 2. Create a shared preprocessing module

Extract duplicated logic from `data_generation.py` and `website/app.py` into a module such as `gesture_preprocessing.py`.

It should provide an interface similar to:

```python
class GestureFeatureBuilder:
    def reset(self): ...
    def add_frame(self, landmarks, handedness, timestamp): ...
    def build_model_input(self, frames, max_frames=93): ...
```

Uploaded-video and live prediction must call the same implementation for:

- Handedness conversion.
- Right/left hand ordering.
- Relative-coordinate calculation.
- Motion-history calculation.
- Padding.
- Velocity augmentation.
- Tensor-shape validation.

The browser should detect landmarks and transmit them. It should not implement model preprocessing.

## 3. Correct video landmark generation

Update `data_generation.py` to:

- Convert OpenCV BGR images to RGB before passing them to MediaPipe:

  ```python
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  ```

- Sort detected hands into explicit right and left slots.
- Keep historical non-wrist landmarks in the same wrist-relative coordinate system used for comparison.
- Remove broad `except:` handling around landmark construction.
- Reset motion history at the beginning of every gesture.
- Use timestamps for motion calculations.
- Use detected-row order for debug indexes instead of source-video frame indexes.

These corrections require regenerating the dataset and retraining the model.

## 4. Replace the live placeholder cache

Keep a variable-length buffer containing only detected frames:

```python
live_frames = []
```

When at least one hand is detected:

```python
features = feature_builder.add_frame(landmarks, handedness, timestamp)
live_frames.append(features)
live_frames = live_frames[-MAX_FRAMES:]
```

When no hand is detected:

- Do not add a model frame.
- Increment the consecutive-silence counter.
- Use silence only to determine when a gesture has ended.

The first detected frame must be appended. It must not update motion history and then be discarded.

## 5. Build padding only in a temporary prediction snapshot

Create the fixed-size input only when prediction or model-input debugging is requested:

```python
def build_model_snapshot(frames):
    raw = np.zeros(
        (MAX_FRAMES, RAW_MODEL_FEATURES),
        dtype=np.float32,
    )

    count = min(len(frames), MAX_FRAMES)
    raw[:count] = frames[-MAX_FRAMES:]

    return add_velocity(raw[np.newaxis, ...])
```

The persistent live buffer contains only real detected frames. Padding exists only in the temporary prediction array.

For example, after 25 detected frames:

```text
Persistent live buffer:  25 real frames
Raw live debug CSV:      25 rows
Temporary model tensor:  25 real rows + 68 zero rows
Model-input debug CSV:   93 rows
```

## 6. Separate live debug artifacts

Write two debug files with distinct purposes:

### `live_gesture_debug.csv`

- Contains only detected frames.
- Has a variable number of rows.
- Includes timestamp and sequential debug index.
- Is used to compare live landmark features with uploaded-video features.

### `live_model_input_debug.csv`

- Contains exactly 93 rows.
- Contains the final padded tensor supplied to the model.
- Is used to inspect padding, velocity augmentation, and final feature layout.

This prevents artificial padding from being mistaken for captured webcam frames.

## 7. Define prediction timing

Use a hybrid prediction strategy:

- Optionally produce preview predictions after a minimum number of detected frames.
- Produce the final prediction after a short consecutive no-hand interval.
- Reset the gesture state after final prediction.

Suggested initial values:

```text
Minimum detected frames: 8-12
Gesture-end silence:     6-10 webcam frames
Maximum gesture frames:  93
```

No-hand frames are segmentation signals and are not model input.

## 8. Repair live lifecycle handling

Update `website/app.py` to:

- Reset `consecutive_empty_frames` whenever a hand is detected.
- Reset feature history when a gesture is completed.
- Clear the gesture buffer after final prediction.
- Prevent motion history from one gesture entering the next.
- Store live state per Socket.IO session rather than in process-wide globals.

Update `website/static/js/index.js` to:

- Emit `reset_live` when the webcam starts.
- Emit `reset_live` when the webcam stops.
- Stop active media tracks when disabling the webcam.
- Send `video.currentTime` as the source-frame timestamp.
- Prevent duplicate `loadeddata` listeners and webcam streams.

## 9. Make padding genuinely maskable

Remove frame index from model features. Recurrent and attention layers already receive frames in sequence order.

Keep timing information outside the model tensor unless experiments show it materially improves validation performance:

```text
Debug metadata       Model features
--------------       --------------
Sequential index     right_present
Timestamp            left_present
                     landmark features
```

With no index in the model tensor, every padding row can be entirely zero and recognized by `Masking(mask_value=0.0)`.

If learned timing is required later, provide an explicit boolean padding mask rather than making padding rows nonzero.

## 10. Clean and rerun the training notebook

Refactor `detect_gestures.ipynb` so that it:

1. Imports the shared preprocessing module.
2. Regenerates all raw gesture features.
3. Pads through the shared tensor builder.
4. Verifies every sample has the expected shape.
5. Verifies every padding row is completely zero.
6. Trains and evaluates a new model version.
7. Saves the model and its preprocessing contract together.

Parameters should include the feature schema, not only tensor dimensions:

```json
{
  "frames_per_gesture": 93,
  "raw_features_per_frame": 254,
  "model_features_per_frame": 508,
  "feature_schema_version": 2,
  "motion_uses_timestamps": true,
  "padding_value": 0.0,
  "hand_order": ["right", "left"]
}
```

The final dimensions must be derived from the implemented schema rather than copied blindly from this example.

## 11. Add parity and regression tests

The primary integration test should replay the same landmark packets through offline and live preprocessing:

1. Capture a browser landmark stream to JSON.
2. Process it through live preprocessing.
3. Process the same packets through the shared offline preprocessing API.
4. Compare the resulting tensors using `numpy.testing.assert_allclose`.

Add focused tests verifying that:

- First-frame motion is zero.
- Right and left hands always occupy the correct slots.
- Detector return order does not affect feature layout.
- No-hand events do not add model rows.
- Ten real frames produce ten data rows and 83 zero rows in a snapshot.
- Reset clears motion history.
- A gesture longer than 93 frames follows the selected truncation/window policy.
- Training and upload preprocessing produce identical tensors for the same video.
- Live and offline preprocessing produce identical tensors for the same landmark stream.

## Implementation order

1. Freeze a v9 baseline and record representative upload/live outputs.
2. Define the feature schema and padding contract.
3. Implement the shared feature builder and unit tests.
4. Migrate `data_generation.py` to the shared builder.
5. Correct BGR-to-RGB conversion and deterministic hand ordering.
6. Replace the live cache with the detected-frame gesture buffer.
7. Add temporary padded snapshots and split debug outputs.
8. Implement gesture segmentation and complete reset behavior.
9. Update browser timestamp and lifecycle events.
10. Regenerate all datasets.
11. Retrain and save a versioned model and parameters contract.
12. Require offline/live tensor parity before measuring browser accuracy.

## Intended live runtime flow

```text
Browser video frame
        |
        v
Are hands detected?
    |             |
   yes            no
    |             |
    v             v
Build one shared   Increment the consecutive
feature frame      silence counter only
    |             |
    v             v
Append it to the   Finalize after the
gesture buffer     configured threshold
    |
    v
Build a temporary 93-row snapshot
    |
    +--> write model-input debug data when enabled
    |
    v
Run preview or final prediction
```

This design gives the model its required fixed input shape without storing artificial empty frames in live state.
