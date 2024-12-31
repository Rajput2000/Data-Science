import cv2
import numpy as np
import math
import os
import yaml
import time
import socket
import smtplib
import subprocess
from email.message import EmailMessage
from flask import Flask, render_template, Response, stream_with_context, request

# Set working directory
os.chdir("/home/Desktop")

def send_ip():
    """
    Send the device's IP address via email to a predefined recipient.

    Uses SMTP to send the IP address of the device obtained from the system.
    """
    hostname = socket.gethostname()
    ip_address = subprocess.getoutput('hostname -I')

    # Set email credentials and recipients
    email_address = "SENDER-EMAIL"
    email_password = "SENDER-PASSWORD"
    
    # Create email message
    msg = EmailMessage()
    msg['Subject'] = "IP ADDRESS"
    msg['From'] = email_address
    msg['To'] = ["RECEIVER-EMAIL"]
    msg.set_content(f"Use this IP address {ip_address}:5000")

    # Send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(email_address, email_password)
        smtp.send_message(msg)
        print("Email sent with IP address.")

# Load YAML configuration for dataset labels
with open('dataset.yaml', mode='r') as f:
    data_yaml = yaml.load(f, Loader=yaml.SafeLoader)
labels = data_yaml['names']

# Load YOLO model
# The model is expected to detect objects in frames captured by the camera or video.
yolo = cv2.dnn.readNetFromONNX('main_best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize video capture
cap = cv2.VideoCapture("./chickens_eating (1080p).mp4")
app = Flask('__name__')

# Main function for object tracking
def main():
    """
    Perform object tracking using YOLO and display annotated frames.

    This function reads frames from a video or camera, processes them using a YOLO
    model for object detection, and tracks identified objects frame-to-frame.
    """
    count = 0
    center_points_prev_frame = []
    tracking_objects = {}
    track_id = 0

    while True:
        ret, image = cap.read()
        count += 1
        if not ret:
            break

        start = time.perf_counter()
        center_points_cur_frame = []

        # Image pre-processing for YOLO input
        row, col, d = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Create a blob for YOLO
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        yolo.setInput(blob)
        preds = yolo.forward()

        # Perform Non-Maximum Suppression
        detections = preds[0]
        boxes, confidences, classes = [], [], []

        # Image dimensions
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        if len(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)) > 0:
            index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

            for ind in index:
                x, y, w, h = boxes_np[ind]
                cx = int((x + x + w) / 2)
                cy = int((y + y + h) / 2)
                center_points_cur_frame.append((cx, cy))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if count <= 2:
            for pt in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    if distance < 20:
                        tracking_objects[track_id] = pt
                        track_id += 1

        else:
            tracking_objects_copy = tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0])
                    if distance < 20:
                        tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                            continue
                if not object_exists:
                    tracking_objects.pop(object_id)

            for pt in center_points_cur_frame:
                tracking_objects[track_id] = pt
                track_id += 1

        for object_id, pt in tracking_objects.items():
            cv2.circle(image, pt, 5, (0, 0, 255), -1)
            cv2.putText(image, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        end = time.perf_counter()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        center_points_prev_frame = center_points_cur_frame.copy()

        key = cv2.waitKey(1)
        if key == 27:
            break

        ret, jpeg = cv2.imencode('.jpg', image)
        img = jpeg.tobytes()

        yield (b' --frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provide video feed via Flask response."""
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the Flask app
app.run(host='0.0.0.0', port='5000', debug=False)
