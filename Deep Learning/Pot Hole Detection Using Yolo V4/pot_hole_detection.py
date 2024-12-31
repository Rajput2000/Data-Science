import cv2
import numpy as np
import requests
import time
from picamera2 import Picamera2

class ObjectDetection:
    """
    A class for performing object detection using YOLOv4 with OpenCV's DNN module.
    
    Attributes:
        nmsThreshold (float): Non-Maximum Suppression threshold.
        confThreshold (float): Confidence threshold for detections.
        image_size (int): The size of the input image for the detection model.
        model (cv2.dnn_DetectionModel): The YOLOv4 detection model.
        classes (list): List of class names for object detection.
        colors (np.ndarray): Random colors for drawing bounding boxes.
    
    Methods:
        __init__(weights_path, cfg_path): Initializes the object detection model.
        load_class_names(classes_path): Loads class names from a file.
        detect(frame): Detects objects in the given frame.
    """

    def __init__(self, weights_path="./yolov4-tiny-custom_best.weights", cfg_path="./yolov4-tiny-custom.cfg"):
        """
        Initialize the object detection model with the specified weights and configuration files.

        Args:
            weights_path (str): Path to the YOLO weights file.
            cfg_path (str): Path to the YOLO configuration file.
        """
        print("Loading Object Detection")
        print("Running OpenCV DNN with YOLOv4")

        self.nmsThreshold = 0.4
        self.confThreshold = 0.7
        self.image_size = 608

        # Load YOLO network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU acceleration
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Initialize the detection model
        self.model = cv2.dnn_DetectionModel(net)
        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        # Set input parameters for the model
        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1 / 255)

    def load_class_names(self, classes_path="./classes_tiny.txt"):
        """
        Load class names from the specified file.

        Args:
            classes_path (str): Path to the file containing class names.

        Returns:
            list: List of class names.
        """
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                self.classes.append(class_name.strip())

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        """
        Perform object detection on the given frame.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            tuple: Detected classes, scores, and bounding boxes.
        """
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

# Initialize the object detection class
ob = ObjectDetection()

# Initialize the Raspberry Pi camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Frame settings
width = 416
height = 416
starting_time = time.time()
frame_counter = 0

# Start capturing and processing frames
while True:
    frame = picam2.capture_array()
    frame_counter += 1

    # Perform object detection
    classes, scores, boxes = ob.detect(frame)

    # Check for specific class detection (e.g., pothole)
    if 0 in classes:
        print('Pothole detected')
        url = "https://api.thingspeak.com/update?api_key=YOUR-THINGSPEAK-API-KEY&field1=0" + str(2)
        response = requests.get(url)
        print(response)
    else:
        url = "https://api.thingspeak.com/update?api_key=YOUR-THINGSPEAK-API-KEY&field1=0" + str(1)
        response = requests.get(url)
        print(response)

    # Draw detection boxes and annotate the frame
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        rec_area = w * h
        frame_area = width * height
        print(ob.classes[classid])

        # Draw boxes for detections meeting criteria
        if len(scores) != 0 and scores[0] >= 0.7:
            if (rec_area / frame_area) <= 0.1 and box[1] < 600:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(frame, f"%{round(scores[0] * 100, 2)} {label}",
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    # Calculate and display FPS
    ending_time = time.time() - starting_time
    fps = frame_counter / ending_time
    cv2.putText(frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('frame', frame)

    # Break on 'q' key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
