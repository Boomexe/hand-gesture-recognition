import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

cwd = os.getcwd()

raw_gesture_data_folder = os.path.join(cwd, "raw_gesture_data")

min_gesture_frame_length = 1
max_length = 0

bad_files = []

for video in os.listdir(raw_gesture_data_folder):
    if video != "4Ll3OtqAzyw":
        continue
    
    video_path = os.path.join(raw_gesture_data_folder, video)
    for filename in os.listdir(video_path):
        file_path = os.path.join(video_path, filename)
        
        df = pd.read_csv(file_path)
        
        if len(df) < min_gesture_frame_length:
            bad_files.append((file_path, video, int(filename.split('.csv')[0])))
            # print(i, file_path)
        
        if len(df) > max_length:
            max_length = len(df)
        
# for path, name, i in bad_files:
#     os.remove(path)

out_df = pd.DataFrame()

for video in os.listdir(raw_gesture_data_folder):
    if video != "4Ll3OtqAzyw":
        continue
    
    words = []
    words_to_delete = []
    with open(f"transcripts/{video}.json", 'r') as transcript:
        transcript = json.load(transcript)
        for word in transcript:
            words.append(word['word'].split(" (")[0].lower())
    
    for path, name, i in bad_files:
        print(len(words))
        print(video, name, i)
        if name == video:
            print(i)
            words_to_delete.append(i)
    
    for index in sorted(words_to_delete, reverse=True):
        words.pop(index)
            
            

    video_path = os.path.join(raw_gesture_data_folder, video)
    # popped_files = 0
    # print(sorted(os.listdir(video_path)))
    ordered_video_path = sorted(os.listdir(video_path), key=lambda x: int(x.split('.csv')[0]))
    print(ordered_video_path)
    for i, filename in enumerate(ordered_video_path):
        # print(filename)
        file_lengths = []
        file_names = []
        
        file_path = os.path.join(video_path, filename)
        
        for other_path, _, p in bad_files:
            if file_path == other_path:
                continue
        
        df = pd.read_csv(file_path)
        if len(df) == 0:
            print(f"{file_path} is empty, skipping")
            # words.pop(i - popped_files)
            # popped_files += 1
            continue
        # if df.tail(1)['frame_index'] >= max_length:
        #     df = df.iloc[0:max_length, 0:]
        
        if len(df) < max_length:
            current_frame = len(df)
            for i in range((max_length - len(df)) // 2):
                # print([current_frame+1, 0] + np.zeros(len(df.columns)-2).tolist())
                # print(len(df.columns), len([current_frame+1, 0] + np.zeros(len(df.columns)-2).tolist()))
                df.loc[len(df)] = [current_frame+1, 0] + np.zeros(len(df.columns)-2).tolist()
                df.loc[len(df)] = [current_frame+1, 1] + np.zeros(len(df.columns)-2).tolist()
                # df = pd.concat([df, pd.DataFrame([current_frame+1, 0] + np.zeros(len(df.columns)-2).tolist(), columns=df.columns)])
                # df = pd.concat([df, pd.DataFrame([current_frame+1, 1] + np.zeros(len(df.columns)-2).tolist(), columns=df.columns)])
        
        df = df.reset_index(drop=True)
        # print(df.columns[df.columns.duplicated()])
        # print(df.head())
        # old: new_df = pd.DataFrame([df.to_numpy().flatten()], columns=[f"{df.at[index, "frame_index"]}_{df.at[index, "hand_index"]}_{col}" for index in df.index for col in df.columns])
        # Use row index instead of frame_index/hand_index to ensure unique column names
        new_df = pd.DataFrame([df.to_numpy().flatten()], columns=[f"row{index}_{col}" for index in df.index for col in df.columns])
        # print(len(new_df))
        # print([f'{i}_{col}' for i, col in enumerate(new_df.columns)])
        # new_df.columns = [f'{i}_{col}' for i, col in enumerate(new_df.columns)]
        # print(out_df.shape, new_df.shape)
        out_df = out_df.reset_index(drop=True)
        new_df = new_df.reset_index(drop=True)
        # print(new_df.columns[new_df.columns.duplicated()])
        # print("123 ", new_df.iloc[0].index.duplicated().any())
        out_df = pd.concat([out_df, new_df.iloc[[0]].reset_index(drop=True)], ignore_index=True)
        # print(out_df.shape)
        # print(out_df)
        
        # print(len(df), len(new_df), len(out_df))
        
        # out_df['word', 0] = words
        # out_df = out_df.iloc[:, 1:]
        # print("sdlkfjsdlkfj ", out_df.shape, len(out_df), len(words))
        # print(out_df)
    out_df.insert(0, 'word', words)
    out_df.to_csv(f"cleaned_gesture_data/{video}.csv", index=False)

