# Hand Gesture Recognition

ASL gesture classification using MediaPipe landmarks and a PyTorch
BiLSTM/attention classifier. Dataset source: [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/).

## Environment

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

## Tests

```powershell
python -m unittest discover -s tests
```

## Training

Use `detect_gestures.ipynb`. It regenerates corrected MediaPipe features,
trains on CUDA when available, evaluates the model, and saves
`hand_gesture_modelv10.pt`. See `RETRAINING_V10.md` for the required cell order
and deployment checks.

The previous TensorFlow notebook is preserved as
`detect_gestures_tensorflow.ipynb`.

## Website

The redesigned **Signbridge** interface and demo-recording checklist are documented
in [WEBSITE_REDESIGN.md](WEBSITE_REDESIGN.md). No retraining is needed for the UI update.

After saving model v10:

```powershell
python website\app.py
```

Live and uploaded-video prediction now send each gesture's exact detected-frame
sequence to the model. Training pads only to the longest gesture in each mini-batch,
and the model mask ignores those temporary padding rows. Live debug CSVs therefore
contain only real detected frames. See `VARIABLE_LENGTH_MIGRATION.md`.
