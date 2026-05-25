import os
import cv2
import mediapipe as mp
import numpy as np
import math

import pandas as pd

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
            "HELLO",
            "THANK YOU",
            "PLEASE",
            "MORE",
            "YES",
            "NO",
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
            found = False
            for word in self.target_words:
                # print(word.casefold()+".mp4", row['path'].casefold())
                if word.casefold()+".mp4" in row['path'].casefold():
                    found = True
                    break
                
            if not found:
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
    
    # read video input and generate landmarks
    def generate(self, video_path):
        with self.HandLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            
            video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            first_frame = 0
            has_changed = False
            
            gesture_data = pd.DataFrame(columns=self.landmark_columns)
        
            last_pos_right = []
            last_pos_left = []
            
            for frame_index in range(0, int(total_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                _, frame = cap.read()
                
                if _:
                    current_timestamp = int((frame_index / video_frame_rate) * 1000)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    result = landmarker.detect_for_video(mp_image, current_timestamp)
                    landmarks = result.hand_landmarks
                    num_hands = len(landmarks)
                    is_first_frame_of_detection = True
                    
                    # NOTE: Does nothing if no hands are detected
                    
                    data = []
                    
                    # if num_hands == 0:
                    #     data.append(frame_index - first_frame)
                    #     data.extend([0,0]) # No hands are present
                    #     data.extend([0.0] * 126) # No landmarks are present
                    
                    
                    for i in range(num_hands):
                        landmark = landmarks[i]
                        current_handedness = result.handedness[i][0].index
                        
                        # Sets the first frame (when gesture starts) to the 0th frame
                        if first_frame == 0 and not has_changed:
                            has_changed = True
                            first_frame = frame_index
                        
                        if i == 0:
                            data.append(frame_index - first_frame)
                            
                            if current_handedness == 0 and num_hands == 1:
                                data.extend([1,0]) # Only right hand is present
                            
                            elif current_handedness == 1 and num_hands == 1:
                                data.extend([0,1]) # Only left hand is present
                            
                            else:
                                data.extend([1,1]) # Right and left hand is present
                        
                        # This contains the data for the current hand
                        nodes = []
                        # This contains blank data (only of frame, current hand, and zeros)
                        extra_nodes = []
                        
                        # SPEED
                        origin_x, origin_y, origin_z = landmark[0].x, landmark[0].y, landmark[0].z
                        
                        
                        if current_handedness == 0 and (len(last_pos_right) == 0):
                            # Absolute position for wrist
                            last_pos_right.append((origin_x, origin_y, origin_z))
                            # Relative position for others
                            last_pos_right.extend((pos.x-origin_x, pos.y-origin_y, pos.z-origin_z) for pos in landmark[1:])
                            
                        elif current_handedness == 1 and (len(last_pos_left) == 0):
                            last_pos_left.append((origin_x, origin_y, origin_z))
                            last_pos_left.extend((pos.x-origin_x, pos.y-origin_y, pos.z-origin_z) for pos in landmark[1:])
                    
                        # elif frame_index - first_frame > 0:                            
                        #     if current_handedness == 0:
                                
                        
                        # if current_handedness == 0: # right hand
                        #     data.append(x_speed)
                        #     data.append(y_speed)
                        #     data.append(z_speed)
                        # speed = get_speed(mp.framework.formats.landmark_pb2.NormalizedLandmark(x=last_x, y=last_y, z=last_z), landmark[0], 1/video_frame_rate)
                        
                        for i, pos in enumerate(landmark):
                            try:
                                # Relative to wrist (origin)
                                rel_x = pos.x - origin_x
                                rel_y = pos.y - origin_y
                                rel_z = pos.z - origin_z
                                
                                if current_handedness == 0:
                                    if i == 0:
                                        speed_x = last_pos_right[i][0] - pos.x
                                        speed_y = last_pos_right[i][1] - pos.y
                                        speed_z = last_pos_right[i][2] - pos.z
                                    else:
                                        speed_x = last_pos_right[i][0] - rel_x
                                        speed_y = last_pos_right[i][1] - rel_y
                                        speed_z = last_pos_right[i][2] - rel_z
                                    
                                    last_pos_right[i] = (pos.x, pos.y, pos.z)
                                
                                elif current_handedness == 1:
                                    if i == 0:
                                        speed_x = last_pos_left[i][0] - pos.x
                                        speed_y = last_pos_left[i][1] - pos.y
                                        speed_z = last_pos_left[i][2] - pos.z
                                    else:
                                        speed_x = last_pos_left[i][0] - rel_x
                                        speed_y = last_pos_left[i][1] - rel_y
                                        speed_z = last_pos_left[i][2] - rel_z
                                    
                                    last_pos_left[i] = (pos.x, pos.y, pos.z)
                                
                                nodes.extend((rel_x, rel_y, rel_z, speed_x, speed_y, speed_z))
                                extra_nodes.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                            
                            except:
                                print("no data")
                        
                        # If there is only the left hand present in frame, we want placeholder frames for the right hand, which is before the left hand in our data
                        if num_hands == 1 and current_handedness == 1:
                            data.extend(extra_nodes)
                            # gesture_data = pd.concat([gesture_data, pd.DataFrame([extra_nodes], columns=landmark_columns)])
                        
                        # Adding data of the current hand (either right or left or both right AND left)
                        data.extend(nodes)
                        # gesture_data = pd.concat([gesture_data, pd.DataFrame([nodes], columns=landmark_columns)])

                        # If there is only the right hand present in frame, we want placeholder frames for the left hand, which is after the right hand in our data
                        if num_hands == 1 and current_handedness == 0:
                            data.extend(extra_nodes)
                            # gesture_data = pd.concat([gesture_data, pd.DataFrame([extra_nodes], columns=landmark_columns)])

                        if is_first_frame_of_detection:
                            is_first_frame_of_detection = False
                    
                    # After looped through both hands
                    # print(len(landmark_columns), len(row))
                    # print(row)
                    if len(data) != 0:
                        gesture_data = pd.concat([gesture_data, pd.DataFrame([data], columns=self.landmark_columns)])
                    
            # print("Done with video: ", row["path"])
            
            # os.makedirs(os.path.join(os.getcwd(), f"asl_citizen_gesture_data/raw_data/{rel_path}/"), exist_ok=True)
            # gesture_data.to_csv(f"asl_citizen_gesture_data/raw_data/{rel_path}/{row['path']}.csv", index=False)
            
            return gesture_data