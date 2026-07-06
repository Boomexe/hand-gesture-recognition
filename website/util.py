# inference_preprocess.py

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


def add_velocity(X):
    velocity = np.diff(X, axis=1, prepend=X[:, :1, :])
    return np.concatenate([X, velocity], axis=-1)


def make_landmark_columns():
    # 21 landmarks * 6 values each * 2 hands + frame_index + right_present + left_present
    cols = ["frame_index", "right_present", "left_present"]

    hand_prefixes = ["right", "left"]
    landmark_names = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]

    for hand in hand_prefixes:
        for lm in landmark_names:
            cols.extend([
                f"{hand}_{lm}_rel_x",
                f"{hand}_{lm}_rel_y",
                f"{hand}_{lm}_rel_z",
                f"{hand}_{lm}_speed_x",
                f"{hand}_{lm}_speed_y",
                f"{hand}_{lm}_speed_z",
            ])

    return cols


def generate_raw_gesture_data(video_path):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    landmark_columns = make_landmark_columns()
    rows = []

    cap = cv2.VideoCapture(video_path)

    video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    first_frame = 0
    has_changed = False

    last_pos_right = []
    last_pos_left = []

    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_index in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()

            if not ok:
                continue

            current_timestamp = int((frame_index / video_frame_rate) * 1000)

            # Keep this the same as training.
            # Training passed the OpenCV frame directly as SRGB, so do the same.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            result = landmarker.detect_for_video(mp_image, current_timestamp)
            landmarks = result.hand_landmarks
            num_hands = len(landmarks)

            data = []

            for hand_i in range(num_hands):
                landmark = landmarks[hand_i]
                current_handedness = result.handedness[hand_i][0].index

                # Your training code treats:
                # 0 = right hand
                # 1 = left hand
                if first_frame == 0 and not has_changed:
                    has_changed = True
                    first_frame = frame_index

                if hand_i == 0:
                    data.append(frame_index - first_frame)

                    if current_handedness == 0 and num_hands == 1:
                        data.extend([1, 0])  # only right
                    elif current_handedness == 1 and num_hands == 1:
                        data.extend([0, 1])  # only left
                    else:
                        data.extend([1, 1])  # both

                nodes = []
                extra_nodes = []

                origin_x = landmark[0].x
                origin_y = landmark[0].y
                origin_z = landmark[0].z

                if current_handedness == 0 and len(last_pos_right) == 0:
                    last_pos_right.append((origin_x, origin_y, origin_z))
                    last_pos_right.extend(
                        (pos.x - origin_x, pos.y - origin_y, pos.z - origin_z)
                        for pos in landmark[1:]
                    )

                elif current_handedness == 1 and len(last_pos_left) == 0:
                    last_pos_left.append((origin_x, origin_y, origin_z))
                    last_pos_left.extend(
                        (pos.x - origin_x, pos.y - origin_y, pos.z - origin_z)
                        for pos in landmark[1:]
                    )

                for lm_i, pos in enumerate(landmark):
                    rel_x = pos.x - origin_x
                    rel_y = pos.y - origin_y
                    rel_z = pos.z - origin_z

                    if current_handedness == 0:
                        if lm_i == 0:
                            speed_x = last_pos_right[lm_i][0] - pos.x
                            speed_y = last_pos_right[lm_i][1] - pos.y
                            speed_z = last_pos_right[lm_i][2] - pos.z
                        else:
                            speed_x = last_pos_right[lm_i][0] - rel_x
                            speed_y = last_pos_right[lm_i][1] - rel_y
                            speed_z = last_pos_right[lm_i][2] - rel_z

                        last_pos_right[lm_i] = (
                            pos.x if lm_i == 0 else rel_x,
                            pos.y if lm_i == 0 else rel_y,
                            pos.z if lm_i == 0 else rel_z,
                        )

                    elif current_handedness == 1:
                        if lm_i == 0:
                            speed_x = last_pos_left[lm_i][0] - pos.x
                            speed_y = last_pos_left[lm_i][1] - pos.y
                            speed_z = last_pos_left[lm_i][2] - pos.z
                        else:
                            speed_x = last_pos_left[lm_i][0] - rel_x
                            speed_y = last_pos_left[lm_i][1] - rel_y
                            speed_z = last_pos_left[lm_i][2] - rel_z

                        last_pos_left[lm_i] = (
                            pos.x if lm_i == 0 else rel_x,
                            pos.y if lm_i == 0 else rel_y,
                            pos.z if lm_i == 0 else rel_z,
                        )

                    nodes.extend([rel_x, rel_y, rel_z, speed_x, speed_y, speed_z])
                    extra_nodes.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                # Match training behavior exactly
                if num_hands == 1 and current_handedness == 1:
                    data.extend(extra_nodes)

                data.extend(nodes)

                if num_hands == 1 and current_handedness == 0:
                    data.extend(extra_nodes)

            if len(data) != 0:
                rows.append(data)

    cap.release()

    df = pd.DataFrame(rows, columns=landmark_columns)
    return df


def preprocess_video_for_model(video_path, max_length, expected_raw_features=None):
    df = generate_raw_gesture_data(video_path)

    if len(df) == 0:
        raise ValueError("No hands were detected in the video.")

    if len(df) > max_length:
        # You can also sample instead of truncating, but truncating is closest to a simple fix.
        df = df.iloc[:max_length].copy()

    if len(df) < max_length:
        current_frame = len(df) - 1

        for _ in range(max_length - len(df)):
            # Match your training padding:
            # [frame_index, 0] + zeros(rest)
            df.loc[len(df)] = [current_frame + 1, 0] + np.zeros(len(df.columns) - 2).tolist()
            current_frame += 1

    df = df.reset_index(drop=True)

    X = df.to_numpy(dtype=np.float32)
    X = X.reshape(1, max_length, X.shape[-1])

    if expected_raw_features is not None and X.shape[-1] != expected_raw_features:
        raise ValueError(
            f"Raw feature mismatch. Got {X.shape[-1]}, expected {expected_raw_features}."
        )

    X = add_velocity(X)

    return X

# import pandas as pd
# import mediapipe as mp
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model

# model = load_model("models/hand_gesture_modelv8.keras")

# def format_landmark_data(landmarks, handednesses):
#     result = []
    
#     # Create a dictionary to store hand data by hand_id
#     hand_data = {}
    
#     # Process detected hands
#     for hand_index in range(len(landmarks)):
#         hand_landmarks = landmarks[hand_index]
#         handedness_info = handednesses[hand_index][0]
#         handedness = handedness_info['categoryName']
#         hand_id = 0 if handedness == "Left" else 1
        
#         hand_data[hand_id] = hand_landmarks
    
#     # Always add both hands (0=Left, 1=Right) in order
#     for hand_id in [0, 1]:
#         result.append(hand_id)
        
#         if hand_id in hand_data:
#             # Hand was detected, add actual coordinates
#             for landmark in hand_data[hand_id]:
#                 result.append(landmark['x'])
#                 result.append(landmark['y'])
#                 result.append(landmark['z'])
#         else:
#             # Hand was not detected, add zeros for all 21 landmarks
#             for _ in range(21):
#                 result.append(0.0)
#                 result.append(0.0)
#                 result.append(0.0)
    
#     # print(len(result))
#     return result

# def format_landmark_data_2d(landmarks, handednesses):
#     """
#     Returns a 2D list: [[left_hand_data], [right_hand_data]]
#     Each hand data includes: [hand_id, x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
#     Missing hands are filled with zeros.
#     """
#     result = [[], []]  # [left_hand, right_hand]
    
#     # Create a dictionary to store hand data by hand_id
#     hand_data = {}
    
#     # Process detected hands
#     for hand_index in range(len(landmarks)):
#         hand_landmarks = landmarks[hand_index]
#         handedness_info = handednesses[hand_index][0]
#         handedness = handedness_info['categoryName']
#         hand_id = 0 if handedness == "Left" else 1
        
#         hand_data[hand_id] = hand_landmarks
    
#     # Build data for both hands
#     for hand_id in [0, 1]:
#         # result[hand_id].append(hand_id)
        
#         if hand_id in hand_data:
#             # Hand was detected, add actual coordinates
#             for landmark in hand_data[hand_id]:
#                 result[hand_id].append(landmark['x'])
#                 result[hand_id].append(landmark['y'])
#                 result[hand_id].append(landmark['z'])
#         else:
#             # Hand was not detected, add zeros for all 21 landmarks
#             for _ in range(21):
#                 result[hand_id].append(0.0)
#                 result[hand_id].append(0.0)
#                 result[hand_id].append(0.0)
    
#     return result  # [[0, x1, y1, z1, ..., x21, y21, z21], [1, x1, y1, z1, ..., x21, y21, z21]]

# def process_video(video_path):
#     # DATA GENERATION. SHOULD ONLY BE RUN TO GENERATE NEW DATA

#     BaseOptions = mp.tasks.BaseOptions
#     HandLandmarker = mp.tasks.vision.HandLandmarker
#     HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
#     HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
#     VisionRunningMode = mp.tasks.vision.RunningMode

#     options = HandLandmarkerOptions(
#         base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
#         running_mode=VisionRunningMode.VIDEO,
#         # delegate=python.BaseOptions.Delegate.GPU,
#         # inference_delegate=python.BaseOptions.Delegate.GPU,
#         # min_hand_presence_confidence=0.75,
#         # min_hand_detection_confidence=0.75,
#         num_hands=2)
        
#     landmarks = ["WRIST_x","WRIST_y","WRIST_z","THUMB_CMC_x","THUMB_CMC_y","THUMB_CMC_z","THUMB_MCP_x","THUMB_MCP_y","THUMB_MCP_z","THUMB_IP_x","THUMB_IP_y","THUMB_IP_z","THUMB_TIP_x","THUMB_TIP_y","THUMB_TIP_z","INDEX_FINGER_MCP_x","INDEX_FINGER_MCP_y","INDEX_FINGER_MCP_z","INDEX_FINGER_PIP_x","INDEX_FINGER_PIP_y","INDEX_FINGER_PIP_z","INDEX_FINGER_DIP_x","INDEX_FINGER_DIP_y","INDEX_FINGER_DIP_z","INDEX_FINGER_TIP_x","INDEX_FINGER_TIP_y","INDEX_FINGER_TIP_z","MIDDLE_FINGER_MCP_x","MIDDLE_FINGER_MCP_y","MIDDLE_FINGER_MCP_z","MIDDLE_FINGER_PIP_x","MIDDLE_FINGER_PIP_y","MIDDLE_FINGER_PIP_z","MIDDLE_FINGER_DIP_x","MIDDLE_FINGER_DIP_y","MIDDLE_FINGER_DIP_z","MIDDLE_FINGER_TIP_x","MIDDLE_FINGER_TIP_y","MIDDLE_FINGER_TIP_z","RING_FINGER_MCP_x","RING_FINGER_MCP_y","RING_FINGER_MCP_z","RING_FINGER_PIP_x","RING_FINGER_PIP_y","RING_FINGER_PIP_z","RING_FINGER_DIP_x","RING_FINGER_DIP_y","RING_FINGER_DIP_z","RING_FINGER_TIP_x","RING_FINGER_TIP_y","RING_FINGER_TIP_z","PINKY_MCP_x","PINKY_MCP_y","PINKY_MCP_z","PINKY_PIP_x","PINKY_PIP_y","PINKY_PIP_z","PINKY_DIP_x","PINKY_DIP_y","PINKY_DIP_z","PINKY_TIP_x","PINKY_TIP_y","PINKY_TIP_z"]
#     landmarks_np = np.array(landmarks).flatten()
#     landmark_columns = landmarks_np.tolist()
#     landmark_columns.insert(0, 'frame_index')
#     landmark_columns.insert(1, 'hand_index')
    
#     cap = cv2.VideoCapture(video_path)
    
#     video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
#     first_frame = 0
#     has_changed = False
    
#     gesture_data = pd.DataFrame(columns=landmark_columns)
    
#     with HandLandmarker.create_from_options(options) as landmarker:
#         for frame_index in range(0, int(total_frames)):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
#             _, frame = cap.read()
            
#             if _:
#                 current_timestamp = int((frame_index / video_frame_rate) * 1000)
#                 mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#                 result = landmarker.detect_for_video(mp_image, current_timestamp)
#                 landmarks = result.hand_landmarks
#                 num_hands = len(landmarks)
                
#                 for i in range(num_hands):
#                     landmark = landmarks[i]
#                     current_handedness = result.handedness[i][0].index
                    
#                     nodes = []
#                     if first_frame == 0 and not has_changed:
#                         has_changed = True
#                         first_frame = frame_index
                    
#                     extra_nodes = []
                    
#                     nodes.append(frame_index - first_frame)
#                     extra_nodes.append(frame_index - first_frame)
#                     # Right: 0, Left: 1
#                     nodes.append(current_handedness)
#                     extra_nodes.append(0 if current_handedness == 1 else 1)
                        
                    
#                     for pos in landmark:
#                         try:
#                             nodes.extend((pos.x, pos.y, pos.z))
#                             extra_nodes.extend((0, 0, 0))
                        
#                         except:
#                             print("no data")
                    
#                     if num_hands == 1 and current_handedness == 1:
#                         gesture_data = pd.concat([gesture_data, pd.DataFrame([extra_nodes], columns=landmark_columns)])
                    
#                     gesture_data = pd.concat([gesture_data, pd.DataFrame([nodes], columns=landmark_columns)])

#                     if num_hands == 1 and current_handedness == 0:
#                         gesture_data = pd.concat([gesture_data, pd.DataFrame([extra_nodes], columns=landmark_columns)])
                        
#         gesture_data.to_csv(f"uploads/processed_landmarks.csv", index=False)
        
#         out_df = pd.DataFrame()

#     df = pd.read_csv("uploads/processed_landmarks.csv")
    
#     max_length = 186
    
#     if len(df) < max_length:
#         current_frame = len(df)
#         for i in range((max_length - len(df)) // 2):
#             df.loc[len(df)] = [current_frame+1, 0] + np.zeros(len(df.columns)-2).tolist()
#             df.loc[len(df)] = [current_frame+1, 1] + np.zeros(len(df.columns)-2).tolist()
    
#     df = df.reset_index(drop=True)
    
#     word = "dummy string"
    
#     new_df = pd.DataFrame([df.to_numpy().flatten()], columns=[f"row{index}_{col}" for index in df.index for col in df.columns])
#     new_df.insert(0, 'word', word)
    
#     out_df = pd.concat([out_df, new_df], ignore_index=True)
    
    
#     out_df.to_csv("uploads/final.csv", index=False)
#     pred = predict()
#     print(pred)
#     print(np.argmax(pred, axis=1)[0])
    
# def predict():
#     df = pd.read_csv("uploads/final.csv")
    
#     X_test = df.loc[:, ~df.columns.str.contains("frame_index|hand_index")].iloc[:, 1:].to_numpy(dtype=np.float32)

#     X_test = X_test.reshape(-1, 186, 63)

#     return model.predict(X_test)