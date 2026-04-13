from dataclasses import dataclass
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np

video_path = "video_files/4Ll3OtqAzyw/0.mp4"
gesture_data_path = "raw_gesture_data/4Ll3OtqAzyw/0.csv"

mp_hands = mp.solutions.hands  # for HAND_CONNECTIONS

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

# -------------------------
# Simple landmark + result types (protobuf-free)
# -------------------------
@dataclass
class SimpleLandmark:
    x: float
    y: float
    z: float

@dataclass
class Category:
    category_name: str

@dataclass
class DetectionResultLike:
    hand_landmarks: list  # list[list[SimpleLandmark]]
    handedness: list      # list[list[Category]]

# -------------------------
# CSV row -> landmarks
# -------------------------
def coords63_to_hand_landmarks(coords63_row_df):
    """
    coords63_row_df: DataFrame slice shape (1, 63)
    returns: list[SimpleLandmark] length 21
    """
    if coords63_row_df is None or coords63_row_df.empty:
        return None

    coords63 = coords63_row_df.iloc[0].tolist()
    if len(coords63) != 63:
        raise ValueError(f"Expected 63 floats, got {len(coords63)}")

    hand = []
    for i in range(21):
        hand.append(SimpleLandmark(
            x=float(coords63[i * 3 + 0]),
            y=float(coords63[i * 3 + 1]),
            z=float(coords63[i * 3 + 2]),
        ))
    return hand

def row_to_detection_result_like(hand_row_df, handedness_name):
    hand = coords63_to_hand_landmarks(hand_row_df)
    if hand is None:
        return None
    return DetectionResultLike(
        hand_landmarks=[hand],
        handedness=[[Category(handedness_name)]]
    )

# -------------------------
# Drawing (manual, no protobuf)
# -------------------------
def draw_hand_cv2(image, hand_landmarks, connections, point_radius=3, line_thickness=2):
    """
    hand_landmarks: list[SimpleLandmark] len 21 with normalized x,y
    connections: iterable of (start_idx, end_idx)
    """
    h, w, _ = image.shape

    # Convert to pixel coords (clamp to image bounds)
    pts = []
    for lm in hand_landmarks:
        x = int(np.clip(lm.x, 0.0, 1.0) * w)
        y = int(np.clip(lm.y, 0.0, 1.0) * h)
        pts.append((x, y))

    # Draw connections first (so points appear on top)
    for a, b in connections:
        cv2.line(image, pts[a], pts[b], (0, 255, 0), line_thickness, cv2.LINE_AA)

    # Draw points
    for (x, y) in pts:
        cv2.circle(image, (x, y), point_radius, (255, 255, 255), -1, cv2.LINE_AA)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Manual draw (no protobuf)
        draw_hand_cv2(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # label
        height, width, _ = annotated_image.shape
        x_coordinates = [lm.x for lm in hand_landmarks]
        y_coordinates = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    return annotated_image

# -------------------------
# Main loop
# -------------------------
cap = cv2.VideoCapture(video_path)
gesture_data = pd.read_csv(gesture_data_path)

play_fps = 30
delay_ms = int(1000 / play_fps)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    gesture_hands = gesture_data[gesture_data["frame_index"] == current_frame_number]

    annotated = frame

    # If your CSV rows are stable order (row0=right, row1=left):
    right_hand_landmarks = gesture_hands.iloc[:1, 2:]
    left_hand_landmarks  = gesture_hands.iloc[1:2, 2:]

    det_right = row_to_detection_result_like(right_hand_landmarks, "Right")
    det_left  = row_to_detection_result_like(left_hand_landmarks, "Left")

    if det_right is not None:
        annotated = draw_landmarks_on_image(annotated, det_right)
    if det_left is not None:
        annotated = draw_landmarks_on_image(annotated, det_left)

    cv2.imshow("Video", annotated)

    if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
