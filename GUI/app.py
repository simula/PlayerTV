from flask import Flask, render_template, request, jsonify, url_for
import subprocess
import os
import uuid
import json
import cv2
import shutil
from tempfile import TemporaryDirectory
from PIL import ImageDraw, ImageFont
from collections import defaultdict 
from PIL import Image as im 


app = Flask(__name__)

@app.route('/')
def index():
    
    return render_template('index.html')


# Global variable to store the path of the last processed MP4 file
last_processed_mp4_path = None

def convert_to_mp4(input_url):
    global last_processed_mp4_path

    output_filename = f"{uuid.uuid4().hex}.mp4"
    output_filepath = os.path.join('static/saved_video', output_filename)
    full_output_path = os.path.join('static', output_filepath)

    if input_url.lower().endswith('.mp4'):
        # If the input is already an MP4, just return its absolute path
        last_processed_mp4_path = os.path.abspath(input_url)
        return url_for('static', filename='saved_video/' + os.path.basename(input_url), _external=True)

    # If the input is M3U8, convert it to MP4
    command = ['ffmpeg', '-i', input_url, '-c:v', 'copy', '-c:a', 'copy', full_output_path]

    try:
        print("Running command:", " ".join(command))
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        last_processed_mp4_path = full_output_path
        return url_for('static', filename='saved_video/' + output_filename, _external=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr.decode('utf-8')}")
        return None
    except subprocess.TimeoutExpired:
        print("FFmpeg command timed out")
        return None

@app.route('/process_video', methods=['POST'])
def process_video():
    input_url = request.json['m3u8Url']
    mp4_path = convert_to_mp4(input_url)
    if mp4_path:
        return jsonify({'mp4Path': mp4_path})
    else:
        return jsonify({'error': 'Failed to process video'}), 500
    


@app.route('/upload_and_filter', methods=['POST'])
def upload_and_filter():
    team = request.form['team'].strip()
    playerId = request.form['playerId'].strip()
    print(f"Received team: {team}, playerId: {playerId}")

    file = request.files['file']

    # Read and parse the JSON file
    json_content = json.load(file.stream)
    frames_data = json_content.get('frame', {})

    # Process the JSON file
    filtered_data = []
    frame_numbers = []
    bounding_boxes = {}
    player_info = {}
    for frame_number, tracks in frames_data.items():
        for track_id, track_info in tracks.items():
            team_value = str(track_info.get('team_id', '')).strip()
            player_id_value = str(track_info.get('kit_number', '')).strip()
            if team_value == team and player_id_value == playerId:
                player_info[int(frame_number)] = {'kit_number': player_id_value}
                filtered_data.append(track_info)
                frame_numbers.append(int(frame_number))
                bounding_boxes.setdefault(int(frame_number), []).append(track_info.get('xywh'))

    # Save the filtered data to a new file
    output_file_path = 'filtered_data.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)

    # Draw bounding boxes on frames
    frames_directory = 'frames'
    shutil.rmtree(frames_directory)
    os.makedirs(frames_directory)
    draw_bounding_boxes(last_processed_mp4_path, frames_directory, filtered_data, player_info)

    # Create video from frames
    output_video_path = os.path.join(app.static_folder, 'videos', 'output_video.mp4')
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    create_video_from_frames(frame_numbers, frames_directory, output_video_path)

    # URL for the video
    video_url = url_for('static', filename='videos/output_video.mp4', _external=True)

    return jsonify({'message': 'File processed successfully', 'videoPath': video_url})

def create_video_from_frames(frames, frames_directory, output_video_path, fps=25):
    with TemporaryDirectory() as temp_dir:
        # Clear the contents of the temporary directory
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Symlink the specific frames into the temporary directory in sequential order
        for i, frame in enumerate(sorted(frames)):
            frame_file = os.path.join(frames_directory, f"{frame}.png")
            temp_frame_file = os.path.join(temp_dir, f"{i+1:04d}.png")
            if os.path.exists(frame_file):
                os.symlink(frame_file, temp_frame_file)
            else:
                print(f"Frame file {frame_file} not found")

        # Prepare and run the ffmpeg command
        frame_pattern = os.path.join(temp_dir, "%04d.png")
        ffmpeg_command = [
            "ffmpeg", "-y", "-i", frame_pattern, "-r", str(fps),
            "-pix_fmt", "yuv420p", output_video_path
        ]

        print("Running ffmpeg command:", " ".join(ffmpeg_command))
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        print(result.stderr.decode())



def draw_bounding_boxes(video_url, frames_directory, bounding_boxes_, player_info):
    _bounding_boxes_ = defaultdict(list)
    for data in bounding_boxes_:
        frame_number, boxes, kit = data.get("frame"), data.get("xywh"), data.get("kit_number")
        _bounding_boxes_[frame_number].append((boxes, kit))

    capture = cv2.VideoCapture(video_url)
    frame_number = 0
    while True:
        frame_path = os.path.join(frames_directory, f"{frame_number}.png")
        ret, image = capture.read()
        if not ret:
            break
        annots = _bounding_boxes_[frame_number]
        frame_number += 1
        if not annots:
            continue

        for annon in annots:
            box, kit = annon
            x, y, w, h = box
            top_left = (int(x), int(y))
            bottom_right = (int(x+w), int(y+h))
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 3)

            kit_number = str(kit)
            print(kit_number, "being drawn on Frame number:", frame_number)

            text_position = (int(x), max(0, int(y) - 5))
            cv2.putText(image, kit_number, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
        cv2.imwrite(frame_path, image)
        


if __name__ == '__main__':
    app.run(debug=True)
