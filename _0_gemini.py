from google import genai
import json
from google.genai import types
import dotenv
import os

dotenv.load_dotenv()

def generate_transcript(video_id):
    client = genai.Client(api_key=os.getenv("API_KEY"))

    # myfile = client.files.upload(file="path/to/sample.mp4"
    # )

    print('Started generating transcript for: ' + video_id)

    # response = client.models.generate_content(
    #     model='models/gemini-2.5-flash',
    #     contents=types.Content(
    #         parts=[
    #             types.Part(
    #                 file_data=types.FileData(
    #                     file_uri=f'https://www.youtube.com/watch?v={video_id}')
    #             ),
    #             types.Part(text='''
    # Please analyze the following video to identify and extract all American Sign Language (ASL) gestures.


    # Output Format Instructions: Do not enclose the output in any markdown, code blocks, or special formatting. The response should be a plain JSON array, nothing more, nothing less. Begin the response directly with [ and end with ].


    # Specific Instructions:


    # Gesture Recognition: For each distinct ASL sign, provide its corresponding English word or phrase. Ensure each entry represents a single, complete gesture from start to finish. Do not combine multiple gestures or include periods of rest.


    # Timestamp Extraction: For each identified gesture, provide the start and end timestamps as a string of hour:min:sec.ms. For example, '01:03:05.35' would represent 1 hour, 3 minutes, 5 seconds and 350 ms.


    # You will use these rules when you provide the following two values for each gesture:
    # timestamp_start: A string representing the precise start time of the ASL gesture from the video's beginning.
    # timestamp_end: A string representing the precise end time of the ASL gesture from the video's beginning.


    # Ensure both values are accurate to the millisecond (e.g., 5.123). The difference between the start and end timestamps should reflect the actual duration of the gesture.


    # Data Format: Present the results as a parsable JSON list object.


    # Example JSON Output:
    # [ { "word": "hello", "timestamp_start": 5.123, "timestamp_end": 6.456 }, { "word": "how are you?", "timestamp_start": "00:02:04.122", "timestamp_end": "00:02:07.18" }, { "word": "thank you", "timestamp_start": "01:00:00.00", "timestamp_end": "01:00:19.00" } ]''')
    #         ]
    #     )
    # )
    
    response = client.models.generate_content(
        model='models/gemini-2.5-flash',
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri=f'https://www.youtube.com/watch?v=4Ll3OtqAzyw')
                ),
                types.Part(text='''
                           
                           Please analyze the following video to identify and extract all American Sign Language (ASL) gestures.


    # Output Format Instructions: Do not enclose the output in any markdown, code blocks, or special formatting. The response should be a plain JSON array, nothing more, nothing less. Begin the response directly with [ and end with ].


    # Specific Instructions:


    # Gesture Recognition: For each distinct ASL sign within each chapter (the sign should match the name of the chapter), provide its corresponding English word or phrase. Ensure each entry represents a single, complete gesture from start to finish. Do not combine multiple gestures or include periods of rest.


    # Timestamp Extraction: For each identified gesture, provide the start and end timestamps as a string of hour:min:sec.ms. For example, '01:03:05.35' would represent 1 hour, 3 minutes, 5 seconds and 350 ms.


    # You will use these rules when you provide the following two values for each gesture:
    # timestamp_start: A string representing the precise start time of the ASL gesture from the video's beginning.
    # timestamp_end: A string representing the precise end time of the ASL gesture from the video's beginning.


    # Ensure both values are accurate to the millisecond (e.g., 5.123). The difference between the start and end timestamps should reflect the actual duration of the gesture.


    # Data Format: Present the results as a parsable JSON list object.
                           
    "chapters": [{"start_time": 86.0, "title": "Don't Want", "end_time": 95.0}, {"start_time": 95.0, "title": "Finish or Done", "end_time": 103.0}, {"start_time": 103.0, "title": "Forget", "end_time": 110.0}, {"start_time": 110.0, "title": "Go", "end_time": 119.0}, {"start_time": 119.0, "title": "Hello", "end_time": 128.0}, {"start_time": 128.0, "title": "Help", "end_time": 141.0}, {"start_time": 141.0, "title": "How are you?", "end_time": 156.0}, {"start_time": 156.0, "title": "Like", "end_time": 166.0}, {"start_time": 166.0, "title": "More", "end_time": 174.0}, {"start_time": 174.0, "title": "Need", "end_time": 181.0}, {"start_time": 181.0, "title": "Nice to meet you", "end_time": 195.0}, {"start_time": 195.0, "title": "No", "end_time": 204.0}, {"start_time": 204.0, "title": "Not, Don't or Doesn't", "end_time": 215.0}, {"start_time": 215.0, "title": "Please", "end_time": 223.0}, {"start_time": 223.0, "title": "Right or Correct", "end_time": 230.0}, {"start_time": 230.0, "title": "Thank you", "end_time": 237.0}, {"start_time": 237.0, "title": "Want", "end_time": 247.0}, {"start_time": 247.0, "title": "Wrong", "end_time": 255.0}, {"start_time": 255.0, "title": "Yes", "end_time": 269.0}, {"start_time": 269.0, "title": "Teaching - Descriptions and Expression", "end_time": 290.0}, {"start_time": 290.0, "title": "Scared", "end_time": 305.0}, {"start_time": 305.0, "title": "Terrified", "end_time": 314.0}, {"start_time": 314.0, "title": "Bad", "end_time": 323.0}, {"start_time": 323.0, "title": "Busy", "end_time": 335.0}, {"start_time": 335.0, "title": "Fine", "end_time": 343.0}, {"start_time": 343.0, "title": "Good or Well", "end_time": 353.0}, {"start_time": 353.0, "title": "Happy", "end_time": 362.0}, {"start_time": 362.0, "title": "Sad", "end_time": 372.0}, {"start_time": 372.0, "title": "Same", "end_time": 383.0}, {"start_time": 383.0, "title": "So-So", "end_time": 391.0}, {"start_time": 391.0, "title": "Teaching", "end_time": 398.0}, {"start_time": 398.0, "title": "ASL Structure vs. English Structure", "end_time": 413.0}, {"start_time": 413.0, "title": "How are you?", "end_time": 425.0}, {"start_time": 425.0, "title": "I'm happy", "end_time": 443.0}, {"start_time": 443.0, "title": "Hello, nice to meet you", "end_time": 458.0}, {"start_time": 458.0, "title": "I need help", "end_time": 474.0}, {"start_time": 474.0, "title": "Yes, same here!", "end_time": 488.0}, {"start_time": 488.0, "title": "You finished", "end_time": 503.0}, {"start_time": 503.0, "title": "Tornado Siren", "end_time": 524.0}, {"start_time": 524.0, "title": "Outro", "end_time": 555}]
    ''')
            ]
        )
    )
    
    print(f"Done generating transcript for {video_id}")

    with open(f'transcripts_text/{video_id}.txt', 'w') as f:
        f.write(response.text)

    data = json.loads(response.text)

    with open(f'transcripts/{video_id}.json', 'w') as f:
        # f.write()
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    video_id = "4Ll3OtqAzyw"
    generate_transcript(video_id)