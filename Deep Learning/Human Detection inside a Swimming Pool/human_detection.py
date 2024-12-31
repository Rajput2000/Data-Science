# File: human_detection.py

import cv2
import numpy as np
import os
import yaml
import time
from yaml.loader import SafeLoader

# Load YAML configuration
with open('./dataset.yaml', mode='r') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)

labels = data_yaml['names']

# Load YOLO model
yolo = cv2.dnn.readNetFromONNX('./best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize video capture
cap = cv2.VideoCapture(0)

count = 0

def main():
    # Initialize count
    count = 0
    center_points_prev_frame = []

    tracking_objects = {}
    track_id = 0

    while True:
        ret, image = cap.read()
        count += 1
        if not ret:
            break

        image = cv2.resize(image, (614, 614))
        start = time.perf_counter()

        # Point current frame
        center_points_cur_frame = []
        row, col, d = image.shape

        # Get YOLO prediction from the image
        # Step 1: Convert image into square image (array)
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Step 2: Get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        yolo.setInput(blob)
        preds = yolo.forward()  # Detection or prediction from YOLO

        # Non-Maximum Suppression
        # Step 1: Filter detection based on confidence (0.4) and probability score (0.25)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # Width and height of the image (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # Confidence of detecting an object
            if confidence > 0.4:
                class_score = row[5:].max()  # Maximum probability from 20 objects
                class_id = row[5:].argmax()  # Get the index position at which max probability occurs

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # Construct bounding from four values: left, top, width, and height
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    # Append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # Clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        if 0 in classes:
            print(count)
            count += 1
            print("Human Detected")

        if count == 2:
            print(count)
            print("Liable to Drown")
            count = 0

        if len(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)) > 0:
            # NMS
            index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

            # Draw the Bounding Box
            for ind in index:
                # Extract bounding box
                x, y, w, h = boxes_np[ind]
                cx = int((x + x + w) / 2)
                cy = int((y + y + h) / 2)
                center_points_cur_frame.append((cx, cy))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        end = time.perf_counter()
        totalTime = end - start
        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow('FRAME', image)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function
main()
