from flask import Flask, render_template, request, jsonify, Response
import os
import cv2


def generate_frames(processed=False):
    global video_source, is_upload
    if not video_source:
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        if processed:
            # Process the frame (apply leak detection bounding box logic here)
            frame = detect_leak(frame)

        # Encode the frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def detect_leak(frame):
    # Add your leak detection logic here
    # For example, detect bounding boxes for leaks and return the processed frame
    return frame