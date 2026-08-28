import os
import json
import flask
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import time

import sys

from flask_socketio import SocketIO, emit
from flask import render_template

sys.path.append("D:/Coding/Projects/hand gesture recognition")
from data_generation import DataGeneration

app = flask.Flask(__name__)

model = load_model("models/hand_gesture_modelv9.keras")

with open("parameters.json", "r") as f:
    parameters = json.load(f)

words = parameters["word_to_number"]

MAX_FRAMES = parameters["frames_per_gesture"]
MODEL_DATA_PER_FRAME = parameters["data_per_frame"]
RAW_DATA_PER_FRAME = MODEL_DATA_PER_FRAME // 2

data_gen = DataGeneration(videos_path="")

is_model_working = False

consecutive_empty_frames = 0
MAX_CONSECUTIVE_EMPTY_FRAMES = 92

data_cache = []

socketio = SocketIO(app, cors_allowed_origins="*")

def add_velocity(X):
    velocity = np.diff(X, axis=1, prepend=X[:, :1, :])
    return np.concatenate([X, velocity], axis=-1)

def preprocess_video_from_upload(video_path):
    df = data_gen.generate(video_path)

    if len(df) == 0:
        raise ValueError("No hands detected in video.")

    if len(df.columns) != RAW_DATA_PER_FRAME:
        raise ValueError(
            f"Raw feature mismatch: got {len(df.columns)}, expected {RAW_DATA_PER_FRAME}. "
            f"Model expects {MODEL_DATA_PER_FRAME} after add_velocity."
        )

    if len(df) > MAX_FRAMES:
        df = df.iloc[:MAX_FRAMES].copy()

    if len(df) < MAX_FRAMES:
        current_frame = len(df) - 1

        for _ in range(MAX_FRAMES - len(df)):
            df.loc[len(df)] = [current_frame + 1, 0] + np.zeros(len(df.columns) - 2).tolist()

    X = df.to_numpy(dtype=np.float32)
    X = X.reshape(1, MAX_FRAMES, RAW_DATA_PER_FRAME)
    X = add_velocity(X)
    # print(X.shape)

    if X.shape[1:] != model.input_shape[1:]:
        raise ValueError(
            f"Final model input mismatch. Got {X.shape}, model expects {model.input_shape}."
        )

    df2 = pd.DataFrame(X.squeeze())
    df2.to_csv("uploaded_video_debug.csv", index=False)
    return X

@app.route("/upload-video", methods=["POST"])
def upload_video():
    if "video" not in flask.request.files:
        return {"message": "No video file uploaded"}, 400

    os.makedirs("uploads", exist_ok=True)

    video_file = flask.request.files["video"]
    video_path = "uploads/uploaded_video.mp4"
    video_file.save(video_path)

    try:
        X = preprocess_video_from_upload(video_path)

        print("X shape:", X.shape)
        print("Model input shape:", model.input_shape)
        print("Nonzero ratio:", np.count_nonzero(X) / X.size)

        pred = model.predict(X)[0]

        top = sorted(
            [(i, float(p)) for i, p in enumerate(pred)],
            key=lambda x: x[1],
            reverse=True
        )

        print("Prediction vector:", pred)
        print("Top predictions:", top[:5])

        predicted_index = int(np.argmax(pred))
        confidence = float(np.max(pred))
        predicted_word = words[str(predicted_index)]

        return {
            "predicted_word": predicted_word,
            "confidence": confidence,
            "class_index": predicted_index,
            "top_predictions": [
                {
                    "class_index": int(i),
                    "word": words[str(i)],
                    "confidence": float(conf),
                }
                for i, conf in top[:5]
            ],
        }, 200

    except Exception as e:
        print("Prediction error:", e)
        return {"message": "Prediction failed", "error": str(e)}, 500

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/by-video")
def video():
    return render_template("by-video.html")


@app.route("/push_landmarks", methods=["POST"])
def push_landmarks():
    data = flask.request.get_json()
    print(data)
    return {"message": "Landmarks received successfully!"}, 200

live_state = {
    "first_frame": None,
    "frame_index": 0,
    "last_pos_right": None,
    "last_pos_left": None,
}

THRESHOLD = 0.8


def reset_live_state():
    data_cache.clear()
    live_state["first_frame"] = None
    live_state["frame_index"] = 0
    live_state["last_pos_right"] = None
    live_state["last_pos_left"] = None


def js_handedness_to_index(handedness):
    """
    MediaPipe JS usually sends categoryName: 'Right' or 'Left'.
    Your Python training code used:
      0 = Right
      1 = Left
    """
    if isinstance(handedness, list):
        handedness = handedness[0]

    label = handedness.get("categoryName") or handedness.get("displayName")

    if label == "Right":
        return 0
    if label == "Left":
        return 1

    # Fallback if JS sends index directly
    return handedness.get("index")


def landmark_tuple(lm):
    return (float(lm["x"]), float(lm["y"]), float(lm["z"]))


def make_hand_features(landmarks, hand_index):
    """
    Returns 21 * 6 = 126 values for one hand:
    rel_x, rel_y, rel_z, speed_x, speed_y, speed_z
    """
    origin_x = float(landmarks[0]["x"])
    origin_y = float(landmarks[0]["y"])
    origin_z = float(landmarks[0]["z"])

    state_key = "last_pos_right" if hand_index == 0 else "last_pos_left"
    last_pos = live_state[state_key]

    current_positions = []

    for i, lm in enumerate(landmarks):
        x = float(lm["x"])
        y = float(lm["y"])
        z = float(lm["z"])

        if i == 0:
            current_positions.append((x, y, z))
        else:
            current_positions.append((x - origin_x, y - origin_y, z - origin_z))

    if last_pos is None:
        # Match training behavior: first detected frame has zero speed.
        last_pos = current_positions.copy()

    features = []

    for i, lm in enumerate(landmarks):
        x = float(lm["x"])
        y = float(lm["y"])
        z = float(lm["z"])

        rel_x = x - origin_x
        rel_y = y - origin_y
        rel_z = z - origin_z

        if i == 0:
            speed_x = last_pos[i][0] - x
            speed_y = last_pos[i][1] - y
            speed_z = last_pos[i][2] - z
        else:
            speed_x = last_pos[i][0] - rel_x
            speed_y = last_pos[i][1] - rel_y
            speed_z = last_pos[i][2] - rel_z

        features.extend([rel_x, rel_y, rel_z, speed_x, speed_y, speed_z])

    live_state[state_key] = current_positions
    return features


def format_live_frame(landmarks_list, handednesses_list):
    """
    Creates one raw frame of length 255:
      frame_index, right_present, left_present,
      126 right-hand features,
      126 left-hand features

    If no hands are detected, returns:
      frame_index, 0, 0,
      126 zeros for right hand,
      126 zeros for left hand
    """
    frame_index = live_state["frame_index"]

    if live_state["first_frame"] is None:
        live_state["first_frame"] = frame_index

    relative_frame_index = frame_index #- live_state["first_frame"]
    live_state["frame_index"] += 1

    right_features = [0.0] * 126
    left_features = [0.0] * 126

    right_present = 0
    left_present = 0

    # Allows empty landmarks/handedness lists
    if landmarks_list and handednesses_list:
        for landmarks, handedness in zip(landmarks_list, handednesses_list):
            hand_index = js_handedness_to_index(handedness)

            if hand_index == 0:
                right_present = 1
                right_features = make_hand_features(landmarks, 0)
            elif hand_index == 1:
                left_present = 1
                left_features = make_hand_features(landmarks, 1)

    row = [relative_frame_index, right_present, left_present]
    row.extend(right_features)
    row.extend(left_features)

    if len(row) != RAW_DATA_PER_FRAME:
        raise ValueError(
            f"Live raw frame mismatch: got {len(row)}, expected {RAW_DATA_PER_FRAME}"
        )

    return row

def predict_live():
    raw = np.array(list(data_cache), dtype=np.float32)

    # Derive frame indexes from cache order so zero-filled placeholders and
    # captured frames share one monotonic timeline without mutating the cache.
    raw[:, 0] = np.arange(len(raw), dtype=np.float32)

    X = raw.reshape(1, MAX_FRAMES, RAW_DATA_PER_FRAME)
    X = add_velocity(X)
    
    df = pd.DataFrame(X.reshape(1, -1).reshape(MAX_FRAMES, MODEL_DATA_PER_FRAME))
    df.to_csv("live_cache_debug.csv", index=False)

    if X.shape[1:] != model.input_shape[1:]:
        raise ValueError(f"Live input mismatch. Got {X.shape}, model expects {model.input_shape}")

    pred = model.predict(X, verbose=0)[0]

    top = sorted(
        [(i, float(p)) for i, p in enumerate(pred)],
        key=lambda x: x[1],
        reverse=True
    )

    predicted_index = int(np.argmax(pred))
    confidence = float(np.max(pred))
    predicted_word = words[str(predicted_index)]

    return {
        "predicted_word": predicted_word,
        "confidence": confidence,
        "class_index": predicted_index,
        "top_predictions": [
            {
                "class_index": int(i),
                "word": words[str(i)],
                "confidence": float(conf),
            }
            for i, conf in top[:5]
        ],
    }
    
    
def find_last_row(cache):
    first_trailing_empty = None

    for i in range(len(cache) - 1, -1, -1):
        # print(frame_is_empty(cache[i][5:]), "checking if frame is empty")
        if frame_is_empty(cache[i][5:]):
            first_trailing_empty = i
        else:
            break

    return first_trailing_empty

# a = [[4, 0, 1, 0, 0], [0, 4, 1, 1, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
# print(find_last_row(a), "last row index aaaaa")

def frame_is_empty(frame) -> bool:
    return all(val == 0 for val in frame)

def empty_frame() -> list:
    return [0.0] * RAW_DATA_PER_FRAME

# def flush_frames()

@socketio.on("landmark_data")
def handle_live_landmarks(data):
    # print(time.time())
    global is_model_working, consecutive_empty_frames, data_cache
    try:
        landmarks = data.get("landmarks", [])
        handednesses = data.get("handednesses", [])
        
        # if len(landmarks) == 0:
        #     emit("prediction", {"status": "no_hands"})
            # return

        frame = format_live_frame(landmarks, handednesses)
        added_this_frame = False
        is_this_frame_empty = frame_is_empty(frame[5:])
        
        if consecutive_empty_frames > MAX_CONSECUTIVE_EMPTY_FRAMES and not frame_is_empty(data_cache[0][5:]):
            data_cache.clear()
            data_cache.append(frame)
            consecutive_empty_frames = 0
            for i in range(MAX_FRAMES-1):
                data_cache.append(empty_frame())
            added_this_frame = True
            print(time.time(), "Clearing cache due to too many empty frames")
            
        if is_this_frame_empty:
            consecutive_empty_frames += 1
        
        # Skips "warming up" phase
        if len(data_cache) == 0 and not added_this_frame:
            for i in range(MAX_FRAMES-1):
                data_cache.append(empty_frame())
            data_cache.append(frame)
            added_this_frame = True
        
        if (len(data_cache) < MAX_FRAMES or is_this_frame_empty and not added_this_frame): # 5 is a random number thats safe i think so we don't check stuff like frame index
            data_cache.append(frame)
            added_this_frame = True
        else:
            last_row_index = find_last_row(data_cache)
            # print(last_row_index, "last row index")
            
            if last_row_index is not None:
                data_cache[last_row_index] = frame
                added_this_frame = True
            
            else:
                data_cache.append(frame)
                added_this_frame = True
        
        
        # if len(data_cache) > MAX_FRAMES:
        while len(data_cache) > MAX_FRAMES:
            data_cache.pop(0)
        
        # print(np.array(data_cache).shape, "frames in cache")
        # print("number of zero frames:", MAX_FRAMES - find_last_row(data_cache))
        
        if len(data_cache) < MAX_FRAMES:
            emit("prediction", {
                "status": "warming_up",
                "frames": len(data_cache),
                "needed": MAX_FRAMES,
            })
            # print(time.time(), "warming up still")
            
            return
        
        if not is_model_working:
            # print(not frame_is_empty(data_cache[0][5:]), "checking if first frame is empty")
            if len(data_cache) >= MAX_FRAMES:
                if not frame_is_empty(data_cache[0][5:]):
                    # print(time.time(), "predicting now")
                    is_model_working = True
                    result = predict_live()
                    result["status"] = "prediction"
                    # print(time.time(), "predicted")
                    is_model_working = False

                    # print("pred time:", time.time())
                    if result["confidence"] >= THRESHOLD:
                        emit("prediction", result)
                    else:
                        emit("prediction", {
                            **result,
                            "status": "low_confidence",
                        })
                
                else:
                    emit("prediction", {
                        "status": "no_hands",
                    })

    except Exception as e:
        print("Live prediction error:", e)
        emit("prediction", {
            "status": "error",
            "error": str(e),
        })


@socketio.on("reset_live")
def handle_reset_live():
    reset_live_state()
    emit("prediction", {"status": "reset"})
    
    

if __name__ == '__main__':
    # pred = predict()
    # print(pred)
    # print(np.argmax(pred, axis=1)[0])
    
    socketio.run(app, host='localhost', port=8000, debug=True)