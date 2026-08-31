# Retraining PyTorch model v10

Model v10 uses the corrected schema-v2 preprocessing and a variable-length
PyTorch classifier. Each gesture retains all of its detected frames. There is no
global frame count and no dataset-level zero padding.

## Environment

Activate the project environment from PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

For a clean environment:

```powershell
python -m pip install -r requirements.txt
```

## What must be regenerated

The existing cleaned JSON files were produced by the old fixed-length cleaning
cell, so regenerate them before training. If the raw split CSVs were already
created with schema v2 (RGB MediaPipe input, deterministic hand slots, and
timestamp-normalized relative motion), you can skip the slow video extraction
cell. Otherwise, regenerate the raw CSVs first.

The existing checkpoint may accept variable-length tensors, but it was trained
from truncated samples. Retraining is required for the model to learn from each
complete gesture and for the deployed artifact to match the new data contract.

## Notebook workflow

Open `detect_gestures.ipynb` from the repository root and run the active cells in
order.

1. Load and normalize the ASL Citizen split metadata.
2. Run the `DataGeneration` cell only when raw schema-v2 CSVs need rebuilding.
3. Run the cleaning cell. It skips empty/unlabelled files and writes every valid
   gesture at its exact detected length without padding or truncation.
4. Load the cleaned JSON and append velocity features independently to each
   gesture.
5. Verify the printed split length ranges and `Training device: cuda`.
6. Train the PyTorch model. `collate_gesture_batch` pads each mini-batch only to
   that batch's longest gesture. Packed LSTMs, masked normalization, attention,
   and pooling ignore those temporary rows.
7. Inspect validation/test accuracy, plots, classification report, and confusion
   matrix.
8. Run the save cell only if performance is acceptable. It writes the checkpoint
   to `models/` and `website/models/`, then updates `website/parameters.json`.
9. Restart the Flask server.

## Run the application

```powershell
python website\app.py
```

Uploaded and live gestures are passed at their exact detected length. The
optional `max_live_frames` parameter is `null` by default; it is only a safety
cap if explicitly configured, not a model shape requirement.

## Tests

```powershell
python -m unittest discover -s tests
```

Tests cover exact-length preprocessing, variable-length batch collation, ignored
batch padding, model output shape, and checkpoint round-tripping.

## Compatibility

Schema v2 retains 255 raw and 510 augmented values per frame. Sequence length is
now dynamic. An old generated JSON dataset or checkpoint should not be presented
as a newly trained variable-length artifact; regenerate cleaned data, retrain,
and deploy the new checkpoint and parameters together.