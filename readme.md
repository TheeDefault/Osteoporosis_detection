# Multi-Class Knee Osteoporosis Classification

This project implements a deep learning pipeline to classify knee X-ray images into three distinct stages of bone health: **Normal**, **Osteopenia**, and **Osteoporosis**. 

It utilizes a pre-trained **EfficientNetV2L** model to extract complex features from high-resolution medical radiographs, achieving a peak accuracy of approximately **92.3%**.

## Dataset
The model is trained on the [Multi-Class Knee Osteoporosis X-Ray Dataset](https://www.kaggle.com/datasets/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset/data) available on Kaggle.

## Key Features
* **Architecture:** `EfficientNetV2L` (Transfer Learning from ImageNet).
* **Image Preprocessing:** 480x480 target resolution with brightness augmentation.
* **Training Optimization:** * `EarlyStopping` to prevent overfitting.
  * `ReduceLROnPlateau` for dynamic learning rate adjustment.
  * `ModelCheckpoint` to save the best-performing weights.

## Requirements
To run this project, install the necessary dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn Pillow tensorflow