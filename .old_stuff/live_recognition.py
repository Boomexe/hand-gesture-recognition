import mediapipe as mp
import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


RESULT = None

model = tf.keras.models.load_model('recognize_gestures_test.keras')

def predict_gesture(hand_landmarks):
    if len(hand_landmarks) == 0:
        print("No hand landmarks detected.")
        return

    for hand in hand_landmarks:
        landmarks = []
        for landmark in hand:
            landmarks.extend([float(landmark.x), float(landmark.y), float(landmark.z)])
        
        landmarks_np = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        # landmarks_np
        print(landmarks_np)
        # x = StandardScaler().fit_transform(landmarks_np)
        pred = model.predict(landmarks_np)
        pred = np.argmax(pred)
        print(pred)
        if 0 <= pred <= 9:
            print(f'Prediction: {pred}')
        
        else:
            print(f'Prediction: {chr(pred+87)}')

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image
# Create a hand landmarker instance with the live stream mode:


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print(result)
    global RESULT
    RESULT = result

def hand_landmarks_from_image(image_path):
    img = cv2.imread(image_path)
    print(img.shape)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    
    image_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE)
    
    with HandLandmarker.create_from_options(image_options) as landmarker:
        result = landmarker.detect(mp_image)
        print(result)
        
        if type(result) is not type(None):
            hand_landmarks_list = result.hand_landmarks
            
            predict_gesture(hand_landmarks_list)
            
            img = draw_landmarks_on_image(img, result)
            
            cv2.imshow('Hand Landmarks', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    
# hand_landmarks_from_image('C:/Users/btamm/Downloads/testhandimage.png')

def hand_landmarks_from_camera():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(1)
        current_frame = 0
        hand_landmarks_list = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.rectangle(frame, (50, 50), (450, 450), (255, 0, 0), 2)
            roi_frame = cv2.cvtColor(frame[50:450, 50:450], cv2.COLOR_BGR2RGB)

            current_frame += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_frame)
            timestamp = int(round(time.time() * 1000))

            if current_frame % 10 == 0:
                landmarker.detect_async(mp_image, timestamp)
                if type(RESULT) is not type(None):
                    hand_landmarks_list = RESULT.hand_landmarks
                    
                    predict_gesture(hand_landmarks_list)
                    
                    frame = draw_landmarks_on_image(frame, RESULT)
                else:
                    pass

            else:
                if hand_landmarks_list is not None:
                    frame = draw_landmarks_on_image(frame, RESULT)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

hand_landmarks_from_camera()