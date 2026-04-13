import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json

from collections import Counter

cwd = os.getcwd()

cleaned_gesture_data_folder = os.path.join(cwd, "cleaned_gesture_data")

df = pd.DataFrame()

for file in os.listdir(cleaned_gesture_data_folder):
    file_path = os.path.join(cleaned_gesture_data_folder, file)
    
    video_df = pd.read_csv(file_path)
    df = pd.concat([df, video_df])
    
words = df['word'].tolist()
word_counts = Counter(words)
print(word_counts)

df.to_csv("final_dataset.csv", index=False)