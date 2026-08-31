import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import flask
import numpy as np
import pandas as pd
import torch
from flask import render_template, request
from flask_socketio import SocketIO, emit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBSITE_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from data_generation import DataGeneration
from gesture_model import load_gesture_checkpoint, select_device
from gesture_preprocessing import (
    RAW_FEATURES_PER_FRAME,
    GestureFeatureBuilder,
    build_model_input,
)

app = flask.Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

with (WEBSITE_ROOT / "parameters.json").open("r", encoding="utf-8") as file:
    parameters = json.load(file)

model_file = parameters.get("model_file", "hand_gesture_modelv10.pt")
model_path = WEBSITE_ROOT / "models" / model_file
model_device = select_device()
if model_path.exists() and model_path.suffix == ".pt":
    model, model_metadata = load_gesture_checkpoint(model_path, model_device)
else:
    model = None
    model_metadata = {}
words = parameters["word_to_number"]
MAX_LIVE_FRAMES = parameters.get("max_live_frames")
MODEL_DATA_PER_FRAME = parameters["data_per_frame"]
RAW_DATA_PER_FRAME = MODEL_DATA_PER_FRAME // 2

if RAW_DATA_PER_FRAME != RAW_FEATURES_PER_FRAME:
    raise ValueError(
        f"Parameter schema expects {RAW_DATA_PER_FRAME} raw values per frame, "
        f"but shared preprocessing produces {RAW_FEATURES_PER_FRAME}."
    )

THRESHOLD = 0.8
MIN_DETECTED_FRAMES = 8
GESTURE_END_EMPTY_FRAMES = 8

data_gen = DataGeneration(videos_path="")


@dataclass
class LiveSessionState:
    feature_builder: GestureFeatureBuilder = field(
        default_factory=GestureFeatureBuilder
    )
    frames: list[list[float]] = field(default_factory=list)
    consecutive_empty_frames: int = 0


live_sessions: dict[str, LiveSessionState] = {}


def get_live_state() -> LiveSessionState:
    return live_sessions.setdefault(request.sid, LiveSessionState())


def reset_live_state(session_id: str) -> None:
    live_sessions.pop(session_id, None)


def prediction_result(model_input: np.ndarray) -> dict:
    if model is None:
        raise RuntimeError(
            f"PyTorch checkpoint not found: {model_path}. Train and save model v10."
        )
    if model_input.shape[2] != model.config.input_size:
        raise ValueError(
            f"Model input mismatch: got {model_input.shape}, "
            f"expected (*, *, {model.config.input_size})."
        )

    inputs = torch.from_numpy(model_input).to(
        device=model_device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        prediction = torch.softmax(model(inputs), dim=1)[0].cpu().numpy()
    top = sorted(
        [(index, float(probability)) for index, probability in enumerate(prediction)],
        key=lambda item: item[1],
        reverse=True,
    )
    predicted_index = int(np.argmax(prediction))

    return {
        "predicted_word": words[str(predicted_index)],
        "confidence": float(prediction[predicted_index]),
        "class_index": predicted_index,
        "top_predictions": [
            {
                "class_index": int(index),
                "word": words[str(index)],
                "confidence": probability,
            }
            for index, probability in top[:5]
        ],
    }


def preprocess_video_from_upload(video_path: str | Path) -> np.ndarray:
    gesture_data = data_gen.generate(str(video_path))
    if len(gesture_data) == 0:
        raise ValueError("No hands detected in video.")
    if len(gesture_data.columns) != RAW_DATA_PER_FRAME:
        raise ValueError(
            f"Raw feature mismatch: got {len(gesture_data.columns)}, "
            f"expected {RAW_DATA_PER_FRAME}."
        )

    raw, model_input = build_model_input(gesture_data.values.tolist())
    pd.DataFrame(raw, columns=data_gen.landmark_columns).to_csv(
        WEBSITE_ROOT / "uploaded_raw_debug.csv", index=False
    )
    pd.DataFrame(model_input[0]).to_csv(
        WEBSITE_ROOT / "uploaded_video_debug.csv", index=False
    )
    return model_input


def build_live_snapshot(frames: list[list[float]]) -> np.ndarray:
    raw, model_input = build_model_input(frames)

    pd.DataFrame(frames, columns=data_gen.landmark_columns).to_csv(
        WEBSITE_ROOT / "live_gesture_debug.csv", index=False
    )
    model_debug = pd.DataFrame(model_input[0])
    model_debug.to_csv(WEBSITE_ROOT / "live_model_input_debug.csv", index=False)
    # Preserve the previous debug filename for existing workflows.
    model_debug.to_csv(WEBSITE_ROOT / "live_cache_debug.csv", index=False)
    return model_input


def emit_prediction_for_frames(frames: list[list[float]], final: bool) -> None:
    result = prediction_result(build_live_snapshot(frames))
    result["status"] = (
        "prediction" if result["confidence"] >= THRESHOLD else "low_confidence"
    )
    result["final"] = final
    result["detected_frames"] = len(frames)
    emit("prediction", result)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/by-video")
def video():
    return render_template("by-video.html")


@app.route("/upload-video", methods=["POST"])
def upload_video():
    if "video" not in flask.request.files:
        return {"message": "No video file uploaded"}, 400

    uploads_path = WEBSITE_ROOT / "uploads"
    uploads_path.mkdir(exist_ok=True)
    video_path = uploads_path / "uploaded_video.mp4"
    flask.request.files["video"].save(video_path)

    try:
        model_input = preprocess_video_from_upload(video_path)
        result = prediction_result(model_input)
        return result, 200
    except Exception as error:
        print("Prediction error:", error)
        return {
            "message": "Prediction failed",
            "error": str(error),
        }, 500


@socketio.on("landmark_data")
def handle_live_landmarks(data):
    state = get_live_state()

    try:
        landmarks = data.get("landmarks", [])
        handednesses = data.get("handednesses", [])

        if landmarks and handednesses:
            state.consecutive_empty_frames = 0

            if (
                MAX_LIVE_FRAMES is None
                or len(state.frames) < int(MAX_LIVE_FRAMES)
            ):
                hands = zip(handednesses, landmarks)
                frame = state.feature_builder.build_frame(
                    hands,
                    frame_index=len(state.frames),
                    timestamp_ms=float(data.get("timestampMs", 0.0)),
                )
                if frame[1] or frame[2]:
                    state.frames.append(frame)

            if len(state.frames) < MIN_DETECTED_FRAMES:
                emit(
                    "prediction",
                    {
                        "status": "warming_up",
                        "frames": len(state.frames),
                        "needed": MIN_DETECTED_FRAMES,
                    },
                )
            else:
                emit_prediction_for_frames(state.frames, final=False)
            return

        if not state.frames:
            emit("prediction", {"status": "no_hands"})
            return

        state.consecutive_empty_frames += 1
        if state.consecutive_empty_frames < GESTURE_END_EMPTY_FRAMES:
            return

        if len(state.frames) >= MIN_DETECTED_FRAMES:
            emit_prediction_for_frames(state.frames, final=True)
        else:
            emit("prediction", {"status": "no_hands"})

        reset_live_state(request.sid)
    except Exception as error:
        print("Live prediction error:", error)
        reset_live_state(request.sid)
        emit(
            "prediction",
            {
                "status": "error",
                "error": str(error),
            },
        )


@socketio.on("reset_live")
def handle_reset_live():
    reset_live_state(request.sid)
    emit("prediction", {"status": "reset"})


@socketio.on("disconnect")
def handle_disconnect():
    reset_live_state(request.sid)


if __name__ == "__main__":
    socketio.run(app, host="localhost", port=8000, debug=True)