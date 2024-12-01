import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
import streamlit as st
import tempfile
import os
import csv
import time

# YOLO model loading
@st.cache_resource
def load_yolo_model():
    return YOLO('yolo11s.pt')

# Define polygon configurations for each video
video_polygon_config = {
    "video1.mp4": {
        "queue_polygon": [[52, 232], [76, 305], [246, 208], [335, 267], [121, 431], [149, 494], [175, 574], [305, 574], [463, 424], [421, 289], [262, 154]],
        "shelf_polygon": [[478, 319], [509, 432], [604, 333], [647, 190], [587, 156]],
    },
    "video2.mp4": {
        "queue_polygon": [[30, 150], [50, 200], [200, 180], [320, 250], [100, 350], [130, 410], [160, 500], [280, 510]],
        "shelf_polygon": [[450, 300], [500, 400], [600, 300], [640, 200], [580, 150]],
    },
    # Add more videos with their respective polygon configurations here...
}

# Global variables for tracking
customer = 0
visitors = 0
tracked_Customers = []
tracked_Visitors = []

def initialize_zones(polygons):
    """Initialize zones based on given polygons."""
    queue_zone = sv.PolygonZone(
        polygon=np.array(polygons["queue_polygon"]),
        triggering_anchors=[sv.Position.CENTER_RIGHT]
    )
    queue_annotator = sv.PolygonZoneAnnotator(zone=queue_zone, color=sv.Color(255, 0, 0))

    shelf_zone = sv.PolygonZone(
        polygon=np.array(polygons["shelf_polygon"]),
        triggering_anchors=[sv.Position.CENTER_LEFT]
    )
    shelf_annotator = sv.PolygonZoneAnnotator(zone=shelf_zone, color=sv.Color(0, 255, 0))

    return queue_zone, queue_annotator, shelf_zone, shelf_annotator

def process_frame(frame, queue_zone, queue_annotator, shelf_zone, shelf_annotator, tracker):
    """Process a single video frame."""
    global customer, visitors, tracked_Customers, tracked_Visitors

    # YOLO detection
    results = yolo_model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Track detections
    tracked_detections = tracker.update_with_detections(detections)
    person_detections = tracked_detections[tracked_detections.class_id == 0]

    # Queue zone
    is_in_queue_zone = queue_zone.trigger(person_detections)
    queue_count = int(np.sum(is_in_queue_zone))

    # Shelf zone
    is_in_shelf_zone = shelf_zone.trigger(person_detections)
    shelf_count = int(np.sum(is_in_shelf_zone))

    # Annotate zones
    annotated_frame = queue_annotator.annotate(scene=frame)
    annotated_frame = shelf_annotator.annotate(scene=annotated_frame)

    return annotated_frame, queue_count, shelf_count

def process_video(video_path, polygons):
    """Process a single video."""
    global customer, visitors, tracked_Customers, tracked_Visitors

    # Initialize zones
    queue_zone, queue_annotator, shelf_zone, shelf_annotator = initialize_zones(polygons)
    tracker = sv.ByteTrack()

    # CSV setup
    csv_file = f'{os.path.basename(video_path).split(".")[0]}_analytics.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Queue Count', 'Shelf Count'])

    # Video setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Unable to open {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create output video file
    processed_video_path = f"processed_{os.path.basename(video_path)}"
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    # Frame processing
    start_time_csv = time.time()
    frame_number = 0  # To track the frame number
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, queue_count, shelf_count = process_frame(frame, queue_zone, queue_annotator, shelf_zone, shelf_annotator, tracker)

        # Write to CSV every 10 seconds
        current_time = time.time()
        if current_time - start_time_csv >= 10:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)),
                    queue_count,
                    shelf_count
                ])
            start_time_csv = current_time

        # Save the processed frame to output video
        out.write(processed_frame)

        frame_number += 1

    cap.release()
    out.release()
    
    # Show processed video in Streamlit
    st.video(processed_video_path)

    st.success(f"Processed video saved as {processed_video_path}")

def main():
    st.title("Multi-Video Processing with YOLOv8")
    st.write("Upload multiple videos to process.")

    # Video upload
    uploaded_files = st.file_uploader("Choose video files", type=["mp4", "avi"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()

            video_name = os.path.basename(uploaded_file.name)
            if video_name in video_polygon_config:
                st.write(f"Processing {video_name}...")
                polygons = video_polygon_config[video_name]
                process_video(tfile.name, polygons)
            else:
                st.error(f"No polygon configuration found for {video_name}")

            os.unlink(tfile.name)

if __name__ == "__main__":
    yolo_model = load_yolo_model()
    main()
