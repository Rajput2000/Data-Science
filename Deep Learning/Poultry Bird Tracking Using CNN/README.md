# Project Documentation

This project demonstrates object detection and tracking using a YOLO-based model, specifically designed for poultry housing applications. It provides a means for the operator to monitor the livestock in the poultry house remotely.

## Prerequisites

- Python 3.7+
- Required Python libraries:
  - OpenCV
  - Numpy
  - Flask
  - PyYAML

## Setup Instructions

1. Clone the repository.
2. Navigate to the project directory.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure the YOLO model (`main_best.onnx`) and the YAML configuration file (`dataset.yaml`) are in the project directory.
5. Replace the email credentials in `send_ip()` with your valid credentials.

## Usage

1. Run the script using the command:

```bash
python main.py
```

2. Access the web application on `http://<your-ip>:5000`.

3. Use the web interface to visualize object tracking in the video stream.

## Features

- Object Detection: Detect objects in video frames using the YOLO model.
- Object Tracking: Assign unique IDs to objects across frames.
- IP Email Notification: Sends the server's IP address for easy accessibility.

## Notes

- Ensure camera/video permissions are granted.
- The script supports video streams as input.

## Known Issues

- Occasional false positives in object detection at lower confidence thresholds.
- Requires manual configuration for email credentials.
