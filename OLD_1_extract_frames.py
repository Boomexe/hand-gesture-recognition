import yt_dlp
from yt_dlp.utils import download_range_func
import os
import re
import cv2
import json
import pandas as pd

base_url = "https://youtube.com/watch?v="

cwd = os.getcwd()
transcripts_folder = os.path.join(cwd, "transcripts")

def custom_filter(row):
    return (row['timestamp_start'], row['timestamp_end'])
    

for filename in os.listdir(transcripts_folder):
    if "4Ll3OtqAzyw" not in filename:
        continue
    
    video_id = os.path.splitext(filename)[0]
    youtube_url = base_url + video_id
    
    df = pd.read_json(os.path.join(transcripts_folder, filename), convert_dates=False)
    
    timestamps = df.apply(custom_filter, axis=1).tolist()
    print(timestamps)

    for i, timestamp in enumerate(timestamps):
        yt_opts = {
            'verbose': True,
            'download_ranges': download_range_func(None, [timestamp]),
            'force_keyframes_at_cuts': True,
            'format': 'best[ext=mp4]',
            'outtmpl': f"video_files/{video_id}/{i}.%(ext)s"
        }

        ydl = yt_dlp.YoutubeDL(yt_opts)
        
        ydl.download(youtube_url)
    # info = list(ydl.extract_info(youtube_url).keys())
    # print(info)

    break
