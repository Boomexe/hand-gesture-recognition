# Changes

## Variable-length gesture migration

- Removed the global gesture-frame count from active preprocessing, training,
  uploaded-video prediction, and live prediction.
- Cleaning now retains every detected frame and performs no padding or
  truncation.
- PyTorch training dynamically pads only within each mini-batch and masks those
  temporary rows throughout the model.
- Live empty frames remain gesture-boundary signals and never enter the feature
  tensor or debug CSV.
- `website/parameters.json` now declares variable-length input and has no default
  live-frame cap.

See `VARIABLE_LENGTH_MIGRATION.md` for implementation details and required
retraining steps.