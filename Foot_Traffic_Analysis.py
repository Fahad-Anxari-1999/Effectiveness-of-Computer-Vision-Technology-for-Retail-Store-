import supervision as sv
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image 
from ultralytics import YOLO
import cv2
import torch
import streamlit as st
import os
import tempfile
import csv
import time
import sys; print(sys.executable)

# YOLO Version 8 Load for face detection
@st.cache_resource
def load_face_model():
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    return YOLO(model_path)

# YOLO Version 8 Load for person detection
@st.cache_resource
def load_yolo_model():
    return YOLO('yolo11s.pt')

# Facial Expression Recognition Model
@st.cache_resource
def load_expression_model():
    processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
    model = AutoModelForImageClassification.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
    return processor, model

face_model = load_face_model()
yolo_model = load_yolo_model()
expression_processor, expression_model = load_expression_model()

# Boundaries allocation
entry_polygon = np.array([[400,580], [680, 250], [800, 500], [350, 780]])
entry_zone = sv.PolygonZone(polygon=entry_polygon, triggering_anchors=[sv.Position.CENTER_RIGHT])
entry_annotator = sv.PolygonZoneAnnotator(entry_zone, color=sv.Color(255,0,0))

shelf1_polygon = np.array([[120,220], [200, 180], [450, 480], [350, 780]])
shelf1_zone = sv.PolygonZone(polygon=shelf1_polygon, triggering_anchors=[sv.Position.CENTER_LEFT])
shelf1_annotator = sv.PolygonZoneAnnotator(shelf1_zone, color=sv.Color(255,0,0))

checkout_polygon = np.array([[400,220], [500, 150],[700, 270], [570, 370]])
checkout_zone = sv.PolygonZone(polygon=checkout_polygon)
checkout_annotator = sv.PolygonZoneAnnotator(checkout_zone, color=sv.Color(255,0,0))

# Initialize counters
total_customer_count = 0
entry_count = 0
shelf1_count = 0
checkout_count = 0
total_customer_enter = 0
total_customer_at_checkout = 0
total_customer_at_shelf1 = 0
tracked_customers = []
tracked_checkout = []
tracked_shelf = []
shelf1_entry_times = {}
shelf1_dwell_times = {}
tracker = sv.ByteTrack()

def process_frame(frame, dwell_times, start_time, zone_entry_counts):
    global total_customer_count, entry_count, shelf1_count, checkout_count, total_customer_enter
    global tracked_customers, total_customer_at_checkout, tracked_checkout
    global total_customer_at_shelf1, tracked_shelf, shelf1_entry_times, shelf1_dwell_times

    results = yolo_model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    tracked_detections = tracker.update_with_detections(detections)
    person_detections = tracked_detections[tracked_detections.class_id == 0]

    is_enter_in_shop = entry_zone.trigger(person_detections)
    customer_count = int(np.sum(is_enter_in_shop))

    is_in_shelf1_zone = shelf1_zone.trigger(person_detections)
    shelf1_zone_count = int(np.sum(is_in_shelf1_zone))

    is_in_checkout_zone = checkout_zone.trigger(person_detections)
    checkout_zone_count = int(np.sum(is_in_checkout_zone))

    for i, (bbox, tracker_id) in enumerate(zip(person_detections.xyxy, person_detections.tracker_id)):
        if is_enter_in_shop[i] or is_in_checkout_zone[0] or is_in_shelf1_zone[0]:
            if tracker_id not in dwell_times:
                dwell_times[tracker_id] = 0
                start_time[tracker_id] = cv2.getTickCount()
                total_customer_count += 1

            if tracker_id in start_time:
                end_tick = cv2.getTickCount()
                elapsed_time = (end_tick - start_time[tracker_id]) / cv2.getTickFrequency()
                dwell_times[tracker_id] = elapsed_time

            if is_enter_in_shop[i] and tracker_id not in tracked_customers:
                tracked_customers.append(tracker_id)
                total_customer_enter += 1

            if is_in_shelf1_zone[i] and tracker_id not in tracked_shelf:
                tracked_shelf.append(tracker_id)
                total_customer_at_shelf1 += 1
                shelf1_entry_times[tracker_id] = cv2.getTickCount()

            if is_in_checkout_zone[i] and tracker_id not in tracked_checkout:
                tracked_checkout.append(tracker_id)
                total_customer_at_checkout += 1

            if tracker_id in shelf1_entry_times:
                entry_tick = shelf1_entry_times.pop(tracker_id)
                time_spent = (cv2.getTickCount() - entry_tick) / cv2.getTickFrequency()
                shelf1_dwell_times[tracker_id] = time_spent

    annotated_frame = entry_annotator.annotate(scene=frame)
    annotated_frame = shelf1_annotator.annotate(scene=annotated_frame)
    annotated_frame = checkout_annotator.annotate(scene=annotated_frame)

    # Display statistics on the frame
    rect_x, rect_y, rect_w, rect_h = frame.shape[1] - 700, frame.shape[0] - 200, 220, 250
    cv2.rectangle(annotated_frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 255, 255), -1)

    cv2.putText(annotated_frame, f'Customers Entry : {customer_count}', (rect_x+10,rect_y+25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
    cv2.putText(annotated_frame, f'Shelf 1 : {shelf1_zone_count}', (rect_x+10, rect_y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),1)
    cv2.putText(annotated_frame, f'Checkout : {checkout_zone_count}', (rect_x+10, rect_y+85), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
    cv2.putText(annotated_frame,f'Total Entry : {total_customer_enter}',(rect_x+10,rect_y+115), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
    cv2.putText(annotated_frame,f'Total Checkouts : {total_customer_at_checkout}',(rect_x+10,rect_y+145), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
    cv2.putText(annotated_frame,f'Shelf1 Totals : {total_customer_at_shelf1}',(rect_x+10,rect_y+175), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
    y_offset = 25
    for tracker_id, time_spent in shelf1_dwell_times.items():
        cv2.putText(annotated_frame, f"ID {tracker_id} - Shelf1 Time: {time_spent:.1f}s", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20

    return annotated_frame, checkout_zone_count, shelf1_zone_count, customer_count

def process_video_stream(video_file):
    csv_file = 'Foot_Traffic_Analysis.csv'
    if not os.path.isfile(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Customer Entry', 'Shelf', 'Checkout','Total Entry', 'Total Checkouts', 'Total Customer at Shelf'])
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.error("Error: Unable to open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_file = "Processed_Foot_Traffic_Analysis.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    image_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, checkout_count, shelf1_count, entry_count = process_frame(frame, {}, {}, {})
        out.write(processed_frame)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        image_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    out.release()
    st.write(f"Processed video saved as {output_file}")

def main2():
    st.title("YOLOv8 + Supervision Video Analysis")
    st.write("Upload a video for facial detection and expression analysis using YOLOv8 and ViT.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        if st.button('Process Video'):
            with st.spinner('Processing video...'):
                process_video_stream(tfile.name)
            st.success('Video processing complete!')

        os.unlink(tfile.name)

if __name__ == "__main__":
    main()
