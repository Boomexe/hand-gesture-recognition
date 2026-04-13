import time
import flask
from flask import json
from flask_socketio import SocketIO, emit
from flask import render_template
from collections import deque
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import mediapipe as mp
import cv2
import os

data_cache = deque(maxlen=186)


app = flask.Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = load_model("models/hand_gesture_modelv6.keras")
words = json.load(open("word_mapping.json", "r"))
length_of_gesture = 7150-2
threshold = 0.8

def format_landmark_data(landmarks, handednesses):
    result = []
    
    # Create a dictionary to store hand data by hand_id
    hand_data = {}
    
    # Process detected hands
    for hand_index in range(len(landmarks)):
        hand_landmarks = landmarks[hand_index]
        handedness_info = handednesses[hand_index][0]
        handedness = handedness_info['categoryName']
        hand_id = 0 if handedness == "Left" else 1
        
        hand_data[hand_id] = hand_landmarks
    
    # Always add both hands (0=Left, 1=Right) in order
    for hand_id in [0, 1]:
        result.append(hand_id)
        
        if hand_id in hand_data:
            # Hand was detected, add actual coordinates
            for landmark in hand_data[hand_id]:
                result.append(landmark['x'])
                result.append(landmark['y'])
                result.append(landmark['z'])
        else:
            # Hand was not detected, add zeros for all 21 landmarks
            for _ in range(21):
                result.append(0.0)
                result.append(0.0)
                result.append(0.0)
    
    # print(len(result))
    return result

def format_landmark_data_2d(landmarks, handednesses):
    """
    Returns a 2D list: [[left_hand_data], [right_hand_data]]
    Each hand data includes: [hand_id, x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
    Missing hands are filled with zeros.
    """
    result = [[], []]  # [left_hand, right_hand]
    
    # Create a dictionary to store hand data by hand_id
    hand_data = {}
    
    # Process detected hands
    for hand_index in range(len(landmarks)):
        hand_landmarks = landmarks[hand_index]
        handedness_info = handednesses[hand_index][0]
        handedness = handedness_info['categoryName']
        hand_id = 0 if handedness == "Left" else 1
        
        hand_data[hand_id] = hand_landmarks
    
    # Build data for both hands
    for hand_id in [0, 1]:
        # result[hand_id].append(hand_id)
        
        if hand_id in hand_data:
            # Hand was detected, add actual coordinates
            for landmark in hand_data[hand_id]:
                result[hand_id].append(landmark['x'])
                result[hand_id].append(landmark['y'])
                result[hand_id].append(landmark['z'])
        else:
            # Hand was not detected, add zeros for all 21 landmarks
            for _ in range(21):
                result[hand_id].append(0.0)
                result[hand_id].append(0.0)
                result[hand_id].append(0.0)
    
    return result  # [[0, x1, y1, z1, ..., x21, y21, z21], [1, x1, y1, z1, ..., x21, y21, z21]]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/by-video")
def video():
    return render_template("by-video.html")

@app.route("/upload-video", methods=["POST"])
def upload_video():
    print(flask.request.files)
    if 'video' not in flask.request.files:
        return {"message": "No video file uploaded"}, 400
    
    video_file = flask.request.files['video']
    video_file.save("uploads/uploaded_video.mp4")
    process_video("uploads/uploaded_video.mp4")
    
    
    
    return {"message": "Video uploaded successfully!"}, 200



def process_video(video_path):
    # DATA GENERATION. SHOULD ONLY BE RUN TO GENERATE NEW DATA

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        # delegate=python.BaseOptions.Delegate.GPU,
        # inference_delegate=python.BaseOptions.Delegate.GPU,
        # min_hand_presence_confidence=0.75,
        # min_hand_detection_confidence=0.75,
        num_hands=2)
        
    landmarks = ["WRIST_x","WRIST_y","WRIST_z","THUMB_CMC_x","THUMB_CMC_y","THUMB_CMC_z","THUMB_MCP_x","THUMB_MCP_y","THUMB_MCP_z","THUMB_IP_x","THUMB_IP_y","THUMB_IP_z","THUMB_TIP_x","THUMB_TIP_y","THUMB_TIP_z","INDEX_FINGER_MCP_x","INDEX_FINGER_MCP_y","INDEX_FINGER_MCP_z","INDEX_FINGER_PIP_x","INDEX_FINGER_PIP_y","INDEX_FINGER_PIP_z","INDEX_FINGER_DIP_x","INDEX_FINGER_DIP_y","INDEX_FINGER_DIP_z","INDEX_FINGER_TIP_x","INDEX_FINGER_TIP_y","INDEX_FINGER_TIP_z","MIDDLE_FINGER_MCP_x","MIDDLE_FINGER_MCP_y","MIDDLE_FINGER_MCP_z","MIDDLE_FINGER_PIP_x","MIDDLE_FINGER_PIP_y","MIDDLE_FINGER_PIP_z","MIDDLE_FINGER_DIP_x","MIDDLE_FINGER_DIP_y","MIDDLE_FINGER_DIP_z","MIDDLE_FINGER_TIP_x","MIDDLE_FINGER_TIP_y","MIDDLE_FINGER_TIP_z","RING_FINGER_MCP_x","RING_FINGER_MCP_y","RING_FINGER_MCP_z","RING_FINGER_PIP_x","RING_FINGER_PIP_y","RING_FINGER_PIP_z","RING_FINGER_DIP_x","RING_FINGER_DIP_y","RING_FINGER_DIP_z","RING_FINGER_TIP_x","RING_FINGER_TIP_y","RING_FINGER_TIP_z","PINKY_MCP_x","PINKY_MCP_y","PINKY_MCP_z","PINKY_PIP_x","PINKY_PIP_y","PINKY_PIP_z","PINKY_DIP_x","PINKY_DIP_y","PINKY_DIP_z","PINKY_TIP_x","PINKY_TIP_y","PINKY_TIP_z"]
    landmarks_np = np.array(landmarks).flatten()
    landmark_columns = landmarks_np.tolist()
    landmark_columns.insert(0, 'frame_index')
    landmark_columns.insert(1, 'hand_index')
    
    cap = cv2.VideoCapture(video_path)
    
    video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    first_frame = 0
    has_changed = False
    
    gesture_data = pd.DataFrame(columns=landmark_columns)
    
    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_index in range(0, int(total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            _, frame = cap.read()
            
            if _:
                current_timestamp = int((frame_index / video_frame_rate) * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = landmarker.detect_for_video(mp_image, current_timestamp)
                landmarks = result.hand_landmarks
                num_hands = len(landmarks)
                
                for i in range(num_hands):
                    landmark = landmarks[i]
                    current_handedness = result.handedness[i][0].index
                    
                    nodes = []
                    if first_frame == 0 and not has_changed:
                        has_changed = True
                        first_frame = frame_index
                    
                    extra_nodes = []
                    
                    nodes.append(frame_index - first_frame)
                    extra_nodes.append(frame_index - first_frame)
                    # Right: 0, Left: 1
                    nodes.append(current_handedness)
                    extra_nodes.append(0 if current_handedness == 1 else 1)
                        
                    
                    for pos in landmark:
                        try:
                            nodes.extend((pos.x, pos.y, pos.z))
                            extra_nodes.extend((0, 0, 0))
                        
                        except:
                            print("no data")
                    
                    if num_hands == 1 and current_handedness == 1:
                        gesture_data = pd.concat([gesture_data, pd.DataFrame([extra_nodes], columns=landmark_columns)])
                    
                    gesture_data = pd.concat([gesture_data, pd.DataFrame([nodes], columns=landmark_columns)])

                    if num_hands == 1 and current_handedness == 0:
                        gesture_data = pd.concat([gesture_data, pd.DataFrame([extra_nodes], columns=landmark_columns)])
                        
        gesture_data.to_csv(f"uploads/processed_landmarks.csv", index=False)
        
        out_df = pd.DataFrame()

    df = pd.read_csv("uploads/processed_landmarks.csv")
    
    max_length = 186
    
    if len(df) < max_length:
        current_frame = len(df)
        for i in range((max_length - len(df)) // 2):
            df.loc[len(df)] = [current_frame+1, 0] + np.zeros(len(df.columns)-2).tolist()
            df.loc[len(df)] = [current_frame+1, 1] + np.zeros(len(df.columns)-2).tolist()
    
    df = df.reset_index(drop=True)
    
    word = "dummy string"
    
    new_df = pd.DataFrame([df.to_numpy().flatten()], columns=[f"row{index}_{col}" for index in df.index for col in df.columns])
    new_df.insert(0, 'word', word)
    
    out_df = pd.concat([out_df, new_df], ignore_index=True)
    
    
    out_df.to_csv("uploads/final.csv", index=False)
    pred = predict()
    print(pred)
    print(np.argmax(pred, axis=1)[0])
    
def predict():
    df = pd.read_csv("uploads/final.csv")
    
    X_test = df.loc[:, ~df.columns.str.contains("frame_index|hand_index")].iloc[:, 1:].to_numpy(dtype=np.float32)

    X_test = X_test.reshape(-1, 186, 63)

    return model.predict(X_test)


# @app.route("/push_landmarks", methods=["POST"])
# def push_landmarks():
#     data = flask.request.get_json()
#     print(data)
#     return {"message": "Landmarks received successfully!"}, 200

@socketio.on('landmark_data')
def handle_landmarks(data):
    # print(f"Received: {data['handednesses']}")
    emit('response', {'status': 'received'})
    print(len(data_cache))
    formatted_landmark = format_landmark_data_2d(data['landmarks'], data['handednesses'])
    data_cache.append(formatted_landmark)
    
    flattened = []
    for i, frame_data in enumerate(data_cache):
        flattened.append(i)
        # flattened.extend(frame_data[0])
        flattened.append(i)
        # flattened.extend(frame_data[1])
    
    # if len(flattened) == 7150:
    #     df = np.array(flattened)
    #     np.savetxt("landmark_data.csv", [df], delimiter=",")
    
    if len(data_cache) == 186:
        try:
            d = pd.DataFrame([flattened])
            d = d.iloc[:, 1:].to_numpy(dtype=np.float32)

            d = d.reshape(-1, 186, 63)
            print(d.shape)
            
            
            pred = model.predict(d)
            if (np.max(pred) > threshold):
                predicted_word = words[str(np.argmax(pred))]
                print(f"Predicted word: {predicted_word} with confidence {np.max(pred)}")
            # print(pred)
            # print(np.argmax(pred, axis=1)[0])
            time.sleep(0.1)  # Simulate processing time
        
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    

if __name__ == '__main__':
    # pred = predict()
    # print(pred)
    # print(np.argmax(pred, axis=1)[0])
    
    socketio.run(app, host='localhost', port=8000, debug=True)

# if __name__ == "__main__":
#     app.run(debug=True)