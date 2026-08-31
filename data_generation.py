import os
import cv2
import mediapipe as mp
import numpy as np
import math

import pandas as pd

from gesture_preprocessing import GestureFeatureBuilder

class DataGeneration:
    def __init__(self, videos_path):
        self.videos_path = videos_path
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        #     print('hand landmarker result: {}'.format(result))

        self.options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=self.VisionRunningMode.VIDEO,
            # delegate=python.BaseOptions.Delegate.GPU,
            # inference_delegate=python.BaseOptions.Delegate.GPU,
            # min_hand_presence_confidence=0.75,
            # min_hand_detection_confidence=0.75,
            num_hands=2)
            
        # landmarks_base = ["WRIST_x","WRIST_y","WRIST_z","THUMB_CMC_x","THUMB_CMC_y","THUMB_CMC_z","THUMB_MCP_x","THUMB_MCP_y","THUMB_MCP_z","THUMB_IP_x","THUMB_IP_y","THUMB_IP_z","THUMB_TIP_x","THUMB_TIP_y","THUMB_TIP_z","INDEX_FINGER_MCP_x","INDEX_FINGER_MCP_y","INDEX_FINGER_MCP_z","INDEX_FINGER_PIP_x","INDEX_FINGER_PIP_y","INDEX_FINGER_PIP_z","INDEX_FINGER_DIP_x","INDEX_FINGER_DIP_y","INDEX_FINGER_DIP_z","INDEX_FINGER_TIP_x","INDEX_FINGER_TIP_y","INDEX_FINGER_TIP_z","MIDDLE_FINGER_MCP_x","MIDDLE_FINGER_MCP_y","MIDDLE_FINGER_MCP_z","MIDDLE_FINGER_PIP_x","MIDDLE_FINGER_PIP_y","MIDDLE_FINGER_PIP_z","MIDDLE_FINGER_DIP_x","MIDDLE_FINGER_DIP_y","MIDDLE_FINGER_DIP_z","MIDDLE_FINGER_TIP_x","MIDDLE_FINGER_TIP_y","MIDDLE_FINGER_TIP_z","RING_FINGER_MCP_x","RING_FINGER_MCP_y","RING_FINGER_MCP_z","RING_FINGER_PIP_x","RING_FINGER_PIP_y","RING_FINGER_PIP_z","RING_FINGER_DIP_x","RING_FINGER_DIP_y","RING_FINGER_DIP_z","RING_FINGER_TIP_x","RING_FINGER_TIP_y","RING_FINGER_TIP_z","PINKY_MCP_x","PINKY_MCP_y","PINKY_MCP_z","PINKY_PIP_x","PINKY_PIP_y","PINKY_PIP_z","PINKY_DIP_x","PINKY_DIP_y","PINKY_DIP_z","PINKY_TIP_x","PINKY_TIP_y","PINKY_TIP_z"]

        self.points = [
            "WRIST",
            "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]

        self.landmarks_base = [
            f"{p}_{s}" for p in self.points for s in ("x", "y", "z", "speed_x", "speed_y", "speed_z")
        ]

        self.right_landmarks = ["right_" + landmark for landmark in self.landmarks_base]
        self.left_landmarks = ["left_" + landmark for landmark in self.landmarks_base]

        self.landmarks = self.right_landmarks + self.left_landmarks

        self.landmarks_np = np.array(self.landmarks).flatten()
        self.landmark_columns = self.landmarks_np.tolist()
        self.landmark_columns.insert(0, 'frame_index')
        # self.landmark_columns.insert(1, 'hand_index')
        self.landmark_columns.insert(1, 'right_present')
        self.landmark_columns.insert(2, 'left_present')

        self.target_words = [
            "CLOSEEYES",
            "EYES",
            "HELLO",
            "MORE",
            "NO",
            "ORANGEEYES",
            "PIANO",
            "PLEASE",
            "RHINO3",
            "SOPHOMORE",
            "THANKYOU",
            "TWOMORE",
            "YES",
        ]
    
    def normalize(self, landmark):
        pass
    
    def get_speed(self, landmark1, landmark2, time_diff):
        if time_diff == 0:
            return 0
        
        x_diff = landmark2.x - landmark1.x
        y_diff = landmark2.y - landmark1.y
        z_diff = landmark2.z - landmark1.z
        
        distance = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        
        speed = distance / time_diff
        
        return speed
    
    def generate_from_dataset(self, csv, rel_path, limit_count=False):
        count = 0
        for i, row in csv.iterrows():
            label = "".join(
                character
                for character in str(row.get("gloss", ""))
                if character.isalnum()
            ).casefold()
            target_words = {word.casefold() for word in self.target_words}
            if label not in target_words:
                continue
            
            # if row['path'] != "4324249049090094-EYES.mp4":
            #     continue
            
            video_path = os.path.join(self.videos_path, row["path"])
            gesture_data = self.generate(video_path)
            
            print("Done with video: ", row["path"])
    
            os.makedirs(os.path.join(os.getcwd(), f"asl_citizen_gesture_data/raw_data/{rel_path}/"), exist_ok=True)
            gesture_data.to_csv(f"asl_citizen_gesture_data/raw_data/{rel_path}/{row['path']}.csv", index=False)
            
            count += 1
    
            if count > 5 and limit_count:
                break
    
    # Read a video and generate one row for each frame containing a detected hand.
    def generate(self, video_path):
        with self.HandLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            if video_frame_rate <= 0:
                cap.release()
                raise ValueError(f"Video has an invalid frame rate: {video_path}")

            feature_builder = GestureFeatureBuilder()
            rows = []

            try:
                for source_frame_index in range(int(total_frames)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
                    frame_read, frame = cap.read()
                    if not frame_read:
                        continue

                    current_timestamp = int(
                        (source_frame_index / video_frame_rate) * 1000
                    )
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    result = landmarker.detect_for_video(
                        mp_image, current_timestamp
                    )

                    if not result.hand_landmarks:
                        continue

                    hands = zip(result.handedness, result.hand_landmarks)
                    rows.append(
                        feature_builder.build_frame(
                            hands,
                            frame_index=len(rows),
                            timestamp_ms=current_timestamp,
                        )
                    )
            finally:
                cap.release()

            return pd.DataFrame(rows, columns=self.landmark_columns)
