import cv2
import os

class VideoProcessor:
    def __init__(self, output_path="output/frames"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def extract_frames(self, video_path, interval=30):
        frames = []
        video = cv2.VideoCapture(video_path)
        frame_count = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if frame_count % interval == 0:
                frames.append(frame)
            frame_count += 1

        video.release()
        return frames
