# Edge-AI System for Real-time Video Analysis

This project implements a real-time video analysis system using Edge AI techniques. It captures video from a camera, processes each frame using a pre-trained MobileNetV2 model, and displays the results in real-time.

## Features

- Real-time video capture and processing
- Object recognition using MobileNetV2 pre-trained on ImageNet
- Display of top prediction and confidence score for each frame
- Real-time FPS (Frames Per Second) calculation and display

## Technologies Used

- Python
- OpenCV for video capture and image processing
- TensorFlow and Keras for the AI model
- NumPy for numerical operations
- Flask for web server (in the hosted version)
