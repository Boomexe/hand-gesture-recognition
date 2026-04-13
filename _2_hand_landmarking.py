print("importing libraries...")
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from mediapipe import solutions
import numpy as np
import cv2
import pandas as pd
import math
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     print('hand landmarker result: {}'.format(result))
    
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)
    # result_callback=print_result)

# video = cv2.VideoCapture("video_files/0.mp4")
dist_folder = "gesture_data/"

# landmarks = [[lm.name+"_x", lm.name+"_y", lm.name+"_z"] for lm in mp.solutions.hands.HandLandmark]
landmarks = ["WRIST_x","WRIST_y","WRIST_z","THUMB_CMC_x","THUMB_CMC_y","THUMB_CMC_z","THUMB_MCP_x","THUMB_MCP_y","THUMB_MCP_z","THUMB_IP_x","THUMB_IP_y","THUMB_IP_z","THUMB_TIP_x","THUMB_TIP_y","THUMB_TIP_z","INDEX_FINGER_MCP_x","INDEX_FINGER_MCP_y","INDEX_FINGER_MCP_z","INDEX_FINGER_PIP_x","INDEX_FINGER_PIP_y","INDEX_FINGER_PIP_z","INDEX_FINGER_DIP_x","INDEX_FINGER_DIP_y","INDEX_FINGER_DIP_z","INDEX_FINGER_TIP_x","INDEX_FINGER_TIP_y","INDEX_FINGER_TIP_z","MIDDLE_FINGER_MCP_x","MIDDLE_FINGER_MCP_y","MIDDLE_FINGER_MCP_z","MIDDLE_FINGER_PIP_x","MIDDLE_FINGER_PIP_y","MIDDLE_FINGER_PIP_z","MIDDLE_FINGER_DIP_x","MIDDLE_FINGER_DIP_y","MIDDLE_FINGER_DIP_z","MIDDLE_FINGER_TIP_x","MIDDLE_FINGER_TIP_y","MIDDLE_FINGER_TIP_z","RING_FINGER_MCP_x","RING_FINGER_MCP_y","RING_FINGER_MCP_z","RING_FINGER_PIP_x","RING_FINGER_PIP_y","RING_FINGER_PIP_z","RING_FINGER_DIP_x","RING_FINGER_DIP_y","RING_FINGER_DIP_z","RING_FINGER_TIP_x","RING_FINGER_TIP_y","RING_FINGER_TIP_z","PINKY_MCP_x","PINKY_MCP_y","PINKY_MCP_z","PINKY_PIP_x","PINKY_PIP_y","PINKY_PIP_z","PINKY_DIP_x","PINKY_DIP_y","PINKY_DIP_z","PINKY_TIP_x","PINKY_TIP_y","PINKY_TIP_z"]
landmarks_np = np.array(landmarks).flatten()
landmark_columns = landmarks_np.tolist()
landmark_columns.insert(0, 'frame_index')
landmark_columns.insert(1, 'hand_index')
# print(landmark_columns)

for youtube_video in os.listdir("video_files/"):
    if youtube_video == ".full_videos":
        continue
    # if youtube_video != "4Ll3OtqAzyw":
    #     continue
    video_path = os.path.join(os.getcwd(), 'video_files', youtube_video)
    print(video_path)
    
    sorted_video_paths = sorted(os.listdir(video_path), key=lambda x: int(x.split('.mp4')[0]))
    print(len(sorted_video_paths))
    for clip_index, file in enumerate(sorted_video_paths):
        full_path = os.path.join(video_path, file)
        clip = cv2.VideoCapture(full_path)
        clip_frame_rate = clip.get(cv2.CAP_PROP_FPS)
        total_frames = clip.get(cv2.CAP_PROP_FRAME_COUNT)

        with HandLandmarker.create_from_options(options) as landmarker:
            gesture_data = pd.DataFrame(columns=landmark_columns)
            
            first_frame = 0
            has_changed = False
            
            for frame_index in range(0, int(total_frames)):
                clip.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                _, frame = clip.read()
                
                if _:
                    current_timestamp = int((frame_index / clip_frame_rate) * 1000)
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
            
            os.makedirs(os.path.join(os.getcwd(), f"raw_gesture_data/{youtube_video}"), exist_ok=True)
            gesture_data.to_csv(f"raw_gesture_data/{youtube_video}/{clip_index}.csv", index=False)