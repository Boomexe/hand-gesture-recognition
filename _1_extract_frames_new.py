import yt_dlp
from yt_dlp.utils import download_range_func
import os
import re
import cv2
import json
import pandas as pd
from moviepy import VideoFileClip, Clip

base_url = "https://youtube.com/watch?v="

cwd = os.getcwd()
transcripts_folder = os.path.join(cwd, "transcripts")
video_files_folder = os.path.join(cwd, "video_files")
full_videos_folder = os.path.join(video_files_folder, ".full_videos")

# NOTE: i get a ValueError: start_time (100.00) should be smaller than the clip's duration (95.34). Check why gemini is giving incorrect timestamps.
#       additionally, gemini can give timestamps differently now because moviepy supports different formats


def custom_filter(row):
    return (row['timestamp_start'], row['timestamp_end'])

def download_video_from_youtube(video_id):
    yt_opts = {
        # 'verbose': True,
        # 'download_ranges': download_range_func(None, [timestamp]),
        # 'force_keyframes_at_cuts': True,
        'format': 'best[ext=mp4]',
        'outtmpl': f"video_files/.full_videos/{video_id}.%(ext)s"
    }

    ydl = yt_dlp.YoutubeDL(yt_opts)

    ydl.download(youtube_url)
    

for filename in os.listdir(transcripts_folder):
    video_id = os.path.splitext(filename)[0]
    if video_id != "4Ll3OtqAzyw":
        continue
    print("only extracting frames from video id:", video_id)
    video_extension = video_id + ".mp4"
    youtube_url = base_url + video_id
    if video_extension not in os.listdir(full_videos_folder):
        download_video_from_youtube(video_id)
    
    transcript = pd.read_json(os.path.join(transcripts_folder, f"{video_id}.json"), dtype=str, convert_dates=False)
    # transcript['timestamp_start'] = pd.to_datetime(transcript['timestamp_start'], format="%H:%M:%S.%f").dt.time
    # transcript['timestamp_end'] = pd.to_datetime(transcript['timestamp_end'], format="%H:%M:%S.%f").dt.time
    # print(transcript.info())
    # Create output directory if it doesn't exist
    
    if video_id not in os.listdir(video_files_folder):
        output_dir = f"video_files/{video_id}"
        os.makedirs(output_dir, exist_ok=True)
        ranges = transcript.apply(custom_filter, axis=1).tolist()
        print(ranges)
        subclip_path = os.path.join(video_files_folder, video_id)
        if len(ranges) != len(os.listdir(subclip_path)):
            clip = VideoFileClip(f"video_files/.full_videos/{video_id}.mp4")

            for i, range in enumerate(ranges):
                subclip = clip.subclipped(range[0], range[1])
                
                subclip.write_videofile(f"video_files/{video_id}/{i}.mp4")