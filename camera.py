from dotenv import load_dotenv

load_dotenv()

import os
import cv2
from datetime import datetime
import settings
import time


FRAMES = 50
SLEEP = 10


def main():
    """
    Capture snippets from video stream
    """

    while True:
        cap = cv2.VideoCapture(settings.VIDEO)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
            time.sleep(SLEEP)
            continue

        frames = 1
        ret, frame = cap.read()

        # initialize video object to record counting
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_path = f"{settings.VIDEO_WRITING_DIRECTORY}{timestamp}.mp4"
        f_height, f_width, _ = frame.shape
        output_video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (f_width, f_height),
        )

        # Read until video is completed
        while cap.isOpened():
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            output_video.write(frame)

            frames += 1
            if frames > FRAMES:
                break

            ret, frame = cap.read()
            if not ret:
                break

        # When everything done, release the video capture object
        cap.release()
        output_video.release()

        # Move completed recording to input directory
        new_output_path = f"{settings.VIDEO_INPUT_DIRECTORY}{timestamp}.mp4"
        os.rename(output_path, new_output_path)

        time.sleep(SLEEP)


if __name__ == "__main__":
    main()
