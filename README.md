# Real-Time Attention Detection System

This project detects a person's attention in real-time using webcam or video input. It leverages OpenCV, MediaPipe, and a Neural Network model to detect:
- Faces and head pose
- Phones in the scene
- The number of hands

The extracted features are then used by a machine learning model to classify whether the person is attentive or not.

## Features
- **Real-time face detection**: Detects faces using MediaPipe Face Detection.
- **Head pose estimation**: Uses MediaPipe Face Mesh to extract facial landmarks and estimate head pose.
- **Phone detection**: Identifies if a person is holding or interacting with a phone.
- **Hand detection**: Detects the number of hands in the frame.
- **Attention classification**: A machine learning model classifies attention based on head pose, face, phone, and hand detection.

## Prerequisites

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- A trained ML model for attention detection (you can replace it with your own model)
-torch>=2.0.0
-torchvision>=0.15.0
-tensorboard>=2.4.1
-ipython
-pycocotools
-thop
-pkg_resources
-pyaml
-tqdm
-requests


## Installation

1. Clone this repository:
```bash
 git clone https://github.com/your-username/attention-detection.git
 cd attention-detection
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) If you don't have a trained attention detection model, follow the instructions in the model's README to train it.
