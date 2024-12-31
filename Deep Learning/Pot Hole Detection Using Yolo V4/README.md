# Pot Hole Detection with YOLOv4 and Raspberry Pi Camera

This project implements object detection using YOLOv4 (Tiny) with OpenCV's DNN module and a Raspberry Pi camera. The primary use case demonstrated here is detecting potholes in real-time and sending detection data to ThingSpeak API.

## Features

- Real-time object detection using YOLOv4 Tiny model.
- GPU acceleration using OpenCV's CUDA backend.
- Integration with Raspberry Pi Camera.
- Pothole detection with automated updates to a ThingSpeak channel.
- Display of detection results with annotated bounding boxes and confidence scores.
- Frames-per-second (FPS) calculation and display.

## Prerequisites

### Hardware Requirements

- Raspberry Pi with a compatible camera module.
- GPU-enabled system (optional but recommended).

### Software Requirements

- Python 3.7+
- OpenCV with CUDA support (if GPU acceleration is required).
- `picamera2` library for Raspberry Pi camera interface.
- Internet connection for ThingSpeak API.

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python dependencies:

   ```bash
   pip install opencv-python numpy requests picamera2
   ```

3. Download the YOLOv4 Tiny weights and configuration files:

   - Place the files `yolov4-tiny-custom_best.weights` and `yolov4-tiny-custom.cfg` in the root directory.

4. Create or update the `classes_tiny.txt` file with the class labels for your detection model.

5. Configure the ThingSpeak API:
   - Replace the API key (`YOUR-THINGSPEAK-API`) in the script with your ThingSpeak channel API key.

## Running the Project

1. Connect the Raspberry Pi camera module and ensure it is properly configured.
2. Run the Python script:
   ```bash
   python pot_hole_detection.py
   ```
3. The camera will start capturing frames, perform object detection, and display the annotated frames in real-time.
4. Press `q` to stop the program.

## Key Files

- `pot_hole_detection.py`: The main script for performing object detection.
- `yolov4-tiny-custom_best.weights`: Pretrained YOLOv4 Tiny weights file (download required).
- `yolov4-tiny-custom.cfg`: YOLOv4 Tiny configuration file (download required).
- `classes_tiny.txt`: Text file containing class labels for the detection model.

## How It Works

1. The script initializes the YOLOv4 Tiny model with the provided weights and configuration files.
2. Frames from the Raspberry Pi camera are processed in real-time.
3. Detected potholes are sent to the ThingSpeak API as updates.
4. Detection results are displayed with bounding boxes, labels, and confidence scores.

## Notes

- Ensure your system has sufficient resources for running real-time object detection.
- For better accuracy, fine-tune the YOLOv4 Tiny model on your dataset.
- Modify the `classes_tiny.txt` file based on your detection needs.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [YOLOv4](https://github.com/AlexeyAB/darknet)
- [ThingSpeak](https://thingspeak.com/)
