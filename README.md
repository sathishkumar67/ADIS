# Animal Intrusion Detection System

This project implements an **Animal Intrusion Detection System** using the state-of-the-art **YOLOv11** object detection models. The system detects animals in real-time from video feeds or images, making it ideal for applications such as:
- Monitoring wildlife in natural habitats
- Preventing animal intrusions in farms or residential areas
- Supporting ecological research and animal behavior studies

The system utilizes three variants of the YOLOv11 model:
- **YOLOv11n (nano)**: Lightweight and fast, perfect for resource-limited environments.
- **YOLOv11s (small)**: Balances speed and accuracy for general use.
- **YOLOv11m (medium)**: Prioritizes higher accuracy for precision-critical applications.

These models are pre-trained and fine-tuned for animal detection, ensuring reliable performance across diverse scenarios.

---

## Table of Contents
- [Introduction](#introduction)
- [YOLOv11 Models](#yolov11-models)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Dataset](#dataset)
- [Examples](#examples)
- [Customization](#customization)
- [Resources](#resources)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

The **Animal Intrusion Detection System** leverages **YOLOv11** models to identify and localize animals in real-time from video streams or static images. Built with Python and the Ultralytics YOLOv11 framework, this system provides an efficient and accurate solution for detecting animal presence, enabling timely alerts or visual outputs for various use cases.

---

## YOLOv11 Models

**YOLOv11** is the latest iteration of the YOLO (You Only Look Once) family, renowned for its real-time object detection capabilities. It offers:
- **Enhanced Accuracy**: Superior detection, especially for smaller objects.
- **Speed Optimization**: Designed for real-time processing.
- **Flexibility**: Supports multiple tasks like detection and segmentation.

This project uses three YOLOv11 variants:
- **YOLOv11n (nano)**: The smallest and fastest, ideal for edge devices.
- **YOLOv11s (small)**: A middle-ground option with good speed and accuracy.
- **YOLOv11m (medium)**: Higher accuracy for demanding applications, with a moderate speed trade-off.

Choose a model based on your specific needs for speed, accuracy, or resource availability.

---

## Setup Instructions

Follow these steps to set up the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/animal-intrusion-detection.git
cd animal-intrusion-detection
