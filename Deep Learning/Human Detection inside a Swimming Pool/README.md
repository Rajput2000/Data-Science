# Human Detection and Tracking in Swimming Pools

This project implements real-time object detection and tracking using a YOLO-based model. It is designed specifically to monitor human activity in swimming pools, focusing on unsafe areas. The system detects humans in these areas and raises alerts if they remain for too long, indicating a potential drowning risk.

## Project Features

- **Real-time Human Detection**: Identifies humans entering unsafe areas of the swimming pool.
- **Object Tracking**: Continuously tracks detected humans across frames.
- **Drowning Risk Alert**: Issues an alert when a human stays too long in an unsafe area, signaling a liability to drown.

## Prerequisites

- Python 3.7+
- Required Python libraries:
  - OpenCV
  - NumPy
  - PyYAML

## Setup Instructions

1. Clone the repository.
2. Navigate to the project directory.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the YOLO model (`main_best.onnx`) and the YAML configuration file (`dataset.yaml`) are in the project directory.

## Usage

1. Run the script using the command:

```bash
python human_detection.py
```

2. The video feed from the camera will be processed in real-time to detect and track humans in unsafe pool areas.

## Notes

- Ensure camera/video permissions are granted.
- The script supports video streams as input.

## Known Issues

- Occasional false positives in object detection at lower confidence thresholds.
- Requires manual configuration for email credentials.
