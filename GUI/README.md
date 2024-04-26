# PlayerTV Flask Application

This Flask web application is designed for processing soccer clips and extracting player tracking data. Its primary function is to enable the tracking and clipping of a specific player within a video clip.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/simula/playertv

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

## Usage

1. Start the flask application:
    ```bash
    python app.py

2. Open your web browser and go to http://localhost:5000/ to access the application.

3. Follow the instructions on the web interface to upload a video file and process it.


## Features

1. Accepts M3U8 HLS playlists as input from the Forzify endpoints for Eliteserien 2023 soccer clips.
2. Processes the video into MP4 format.
3. Accepts a JSON file containing processed information about all players, including their identification, re-identification and tracking data.
4. Provides users with the ability to select teams and players involved in the soccer clip.
5. Initiates clipping of the video into frames showcasing the specified player based on user selection.
6. Displays the clipped video as output.

## License
This project is licensed under the MIT License.