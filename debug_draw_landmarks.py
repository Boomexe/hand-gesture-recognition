from dataclasses import dataclass
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np

# vid = "4324249049090094-EYES.mp4"
vid = "08323987948608247-TWO MORE.mp4"
video_path = f"asl citizen data/ASL_Citizen/videos/{vid}"
# gesture_data_path = f"asl_citizen_gesture_data/raw_data/test/{vid}.csv"
gesture_data_path = f"website/uploads/processed_landmarks.csv"

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# ChatGPT stuff
@dataclass
class SimpleLandmark:
    x: float
    y: float
    z: float

def coords63_to_hand_landmarks(coords63, handedness):
    """
    coords63: iterable of length 63:
      [WRIST_x, WRIST_y, WRIST_z, THUMB_CMC_x, ..., PINKY_TIP_z]
    returns: list[SimpleLandmark] length 21 (what mp_drawing.draw_landmarks can draw)
    """
    
    if (coords63.empty):
        return [], None
    coords63 = list(coords63.iloc[0])

    if len(coords63) != 63:
        raise ValueError(f"Expected 63 floats, got {len(coords63)}")

    hand = []
    # print(coords63)
    for i in range(21):
        hand.append(SimpleLandmark(
            x=float(coords63[i*3 + 0]),
            y=float(coords63[i*3 + 1]),
            z=float(coords63[i*3 + 2]),
        ))

    return hand, handedness

def draw_landmarks_on_image(rgb_image, detection_result):
    for landmark in detection_result[0]:
        x, y = int(landmark.x * rgb_image.shape[1]), int(landmark.y * rgb_image.shape[0])
        
        rgb_image = cv2.circle(rgb_image, (x, y), 5, (0, 255, 0), -1)
    
    return rgb_image

cap = cv2.VideoCapture(video_path)

gesture_data = pd.read_csv(gesture_data_path)

play_fps = 30  # playback speed you WANT (lower = slower)
delay_ms = int(1000 / play_fps)

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        gesture_hands = gesture_data[gesture_data['frame_index'] == current_frame_number]
        
        right_hand_landmarks = gesture_hands.iloc[:1, 2:]
        left_hand_landmarks = gesture_hands.iloc[1:2, 2:]
        
        det = coords63_to_hand_landmarks(right_hand_landmarks, 0)
        det2 = coords63_to_hand_landmarks(left_hand_landmarks, 1)
        
        annotated = draw_landmarks_on_image(frame, det)
        annotated = draw_landmarks_on_image(annotated, det2)
        
        annotated = cv2.resize(annotated, (960, 540))
        
        cv2.imshow("Video", annotated)
        
        
        draw_landmarks_on_image
    
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break
    
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

cap.release()
cv2.destroyAllWindows()