import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import cv2
import pandas as pd

import os

cwd = os.getcwd()
gestures = os.path.join(cwd, 'gestures')
train_dataset = os.path.join(gestures, 'train')
test_dataset = os.path.join(gestures, 'test')

new_gestures = os.path.join(cwd, 'new_gestures')
new_test_dataset = os.path.join(new_gestures, 'test')
new_train_dataset = os.path.join(new_gestures, 'train')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
LandmarkEnum = mp.solutions.hands.HandLandmark
LandmarkNames = [landmark.name for landmark in LandmarkEnum]
LandmarkNamesExpanded = [[landmark.name + "x", landmark.name + "y", landmark.name + "z"] for landmark in LandmarkEnum]
LandmarkNamesExpanded = np.array(LandmarkNamesExpanded).flatten().tolist()
LandmarkNamesExpanded.append('TARGET')

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

# for category in os.listdir(train_dataset):
#     os.mkdir(os.path.join(new_train_dataset, category))
#     for file in os.listdir(os.path.join(train_dataset, category)):
#         image = cv2.imread(os.path.join(train_dataset, category, file))
#         image = cv2.copyMakeBorder(image, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         cv2.imwrite(os.path.join(new_train_dataset, category, file[:-5] + '.png'), image)


with HandLandmarker.create_from_options(options) as landmarker:
    hand_df = pd.DataFrame(columns=LandmarkNamesExpanded)
    hand_df_test = pd.DataFrame(columns=LandmarkNamesExpanded)

    # for category in os.listdir(new_train_dataset):
    #     for file in os.listdir(os.path.join(new_train_dataset, category)):
    #         mp_image = mp.Image.create_from_file(os.path.join(new_train_dataset, category, file))

    #         result = landmarker.detect(mp_image)

    #         for hand in result.hand_landmarks:
    #             landmarks = []

    #             for i, landmark in enumerate(hand):
    #                 landmarks.extend([landmark.x, landmark.y, landmark.z])
                
    #             landmarks.append(category)
    #             hand_df.loc[hand_df.shape[0]] = landmarks

    #     print('finished category: {}'.format(category))
    
    for category in os.listdir(new_test_dataset):
        for file in os.listdir(os.path.join(new_test_dataset, category)):
            mp_image = mp.Image.create_from_file(os.path.join(new_test_dataset, category, file))

            result = landmarker.detect(mp_image)

            for hand in result.hand_landmarks:
                landmarks = []

                for i, landmark in enumerate(hand):
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks.append(category)
                hand_df.loc[hand_df.shape[0]] = landmarks

        print('finished category: {}'.format(category))
    
    # for category in os.listdir(new_test_dataset):
    #     # os.mkdir(os.path.join(cwd, 'new_gestures', 'test', category), exist_ok=True)
    #     for file in os.listdir(os.path.join(new_test_dataset, category)):
    #         # image = cv2.imread(os.path.join(test_dataset, category, file))
    #         # image = cv2.copyMakeBorder(image, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #         # cv2.imshow('r', image)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
    #         # cv2.imwrite(os.path.join(cwd, 'new_gestures', 'test', category, file[:-5] + '.png'), image)
    #         # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    #         mp_image = mp.Image.create_from_file(os.path.join(new_test_dataset, category, file))

    #         result = landmarker.detect(mp_image)
    #         # if len(result.hand_landmarks) != 0:
    #         #     print(os.path.join(test_dataset, category, file))
    #         print(len(result.hand_landmarks), file)
    #         for hand in result.hand_landmarks:
    #             landmarks = []

    #             for i, landmark in enumerate(hand):
    #                 landmarks.extend([landmark.x, landmark.y, landmark.z])
                
    #             landmarks.append(category)
    #             hand_df_test.loc[hand_df_test.shape[0]] = landmarks

    #     print('finished category: {}'.format(category))

# hand_df.to_csv(os.path.join(cwd, 'train.csv'), index=True)
hand_df.to_csv(os.path.join(cwd, 'test.csv'), index=True)