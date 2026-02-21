# 🛣️ Pothole Detection using YOLOv8 (Deep Learning)

This project implements an **end-to-end pothole detection system** using **YOLOv8 (Convolutional Neural Network–based object detection)**.  
The model detects potholes in road images and highlights them using bounding boxes with confidence scores.


## 🚀 Project Highlights
- Deep Learning–based **real-world problem**
- Uses **YOLOv8 (CNN architecture)**
- Custom annotated dataset
- Trained and tested on **Google Colab (GPU)**
- Supports **image-based inference**


## 🧠 Model Details
- **Model**: YOLOv8 Nano
- **Framework**: Ultralytics YOLO
- **Input Size**: 640 × 640
- **Epochs**: 50
- **Task**: Object Detection
- **Classes**: Pothole


## 📂 Project Structure

path_hole_detection/
│
├── model_train.ipynb # Training notebook
├── inference.py # Inference script
├── requirements.txt # Dependencies
├── README.md # Project documentation
│
├── weights/
│ └── best.pt # Trained YOLOv8 model
│
├── sample_inputs/
│ └── test.jpg # Sample test image
│
└── outputs/
└── predictions/ # Inference results


## 🏋️ Model Training
Training was performed using the Ultralytics YOLOv8 framework with GPU acceleration on Google Colab.

Key steps:
1. Dataset prepared in YOLO format
2. Data split into training and validation sets
3. Model trained for 50 epochs
4. Best model saved as `best.pt`


## 🔍 Inference (How to Run)
Follow these steps to test pothole detection:

```bash
pip install ultralytics
python inference.py