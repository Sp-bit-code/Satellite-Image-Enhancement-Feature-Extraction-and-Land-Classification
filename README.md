# Satellite-Image-Enhancement-Feature-Extraction-and-Land-Classification



# Satellite Image Land Classification System

## 1. Overview

This project is a deep learning-based system that classifies satellite images into different land categories such as Forest, River, SeaLake, Residential, and Industrial.

It integrates image preprocessing, enhancement, and classification into a single pipeline. A Streamlit-based interface allows users to upload images and get real-time predictions with confidence scores.

---

## 2. Project Architecture

```
satellite_image_land_classification/
│
├── app.py
├── model/
│   ├── land_classifier_model.keras
│   └── class_labels.json
│
├── utils/
│   ├── classification.py
│   ├── image_enhancement.py
│   └── feature_extraction.py
│
├── data/
│   ├── sample_images/
│   └── outputs/
│
├── requirements.txt
└── README.md
```

---

## 3. Model Details

### Model Type

* Convolutional Neural Network (CNN)
* Based on MobileNetV2 architecture

### Input Specifications

* Input Size: 224 × 224 × 3
* Color Format: RGB
* Normalization: Pixel values scaled to [0, 1]

### Output

* Softmax classification layer
* Output shape: (1, 10)

### Classes

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

### Training Details

* Dataset: EuroSAT
* Loss Function: Categorical Crossentropy
* Optimizer: Adam
* Metric: Accuracy

### Important Notes (Critical for Accuracy)

* Preprocessing during inference must match training
* Always use RGB format
* Normalize image (divide by 255)
* Do not use enhanced image for prediction
* Ensure label order is correct

---

## 4. Tech Stack

### Language

* Python

### Libraries

* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* PIL

---

## 5. Installation

### Install all dependencies

```
pip install tensorflow opencv-python numpy streamlit pillow matplotlib scikit-learn
```

---

## 6. How to Run

### Method 1: Normal

```
streamlit run app.py
```

---

### Method 2: Using .bat File (Windows)

Create file: `run_app.bat`

```
@echo off
cd /d %~dp0
streamlit run app.py
pause
```

Double click to run.

---

### Method 3: Using Python module

```
python -m streamlit run app.py
```

---

### Method 4: VS Code

* Open folder
* Open terminal
* Run:

```
streamlit run app.py
```

---

## 7. System Workflow

1. User uploads image
2. Image is validated
3. Resized to model size
4. Converted to RGB
5. Normalized
6. Passed into CNN
7. Predictions generated
8. Results displayed

---

## 8. Image Enhancement Module

Used for improving image quality (not for model input)

### Techniques:

* CLAHE
* Denoising
* Sharpening
* Gamma correction
* Histogram processing

---

## 9. Syllabus Coverage

### Digital Image Formation and Low-Level Processing

* Image representation
* Image transformation
* Convolution and filtering
* Image enhancement
* Histogram processing

---

### Depth Estimation and Multi-Camera Views

* Perspective transformation
* Geometric understanding of images
* Basic spatial transformations

---

### Feature Extraction and Image Segmentation

* Edge detection concepts
* Texture understanding
* CNN-based automatic feature extraction
* Scale-space concepts

---

### Pattern Analysis and Motion Analysis

* Classification using supervised learning
* CNN (Artificial Neural Networks)
* Probability-based prediction
* Accuracy evaluation
* Confusion matrix

---

### Shape from X

* Light and surface interaction
* Texture and color-based classification
* Reflectance understanding

---

## 10. Common Issues and Fixes

### Issue: Same prediction (SeaLake)

**Reasons:**

* Wrong preprocessing
* Label mismatch
* Model overfitting
* Using enhanced image

**Fix:**

* Use RGB format
* Normalize properly
* Use original image
* Check label order

---

### Issue: Model not loading

**Fix:**

* Use `.keras` format
* Match TensorFlow version
* Handle custom layers properly

---

## 11. Applications

* Environmental monitoring
* Urban planning
* Agriculture analysis
* Water detection
* Remote sensing

---

## 12. Future Improvements

* Improve model accuracy
* Add more classes
* Add Grad-CAM visualization
* Deploy on cloud
* Integrate satellite APIs

---

## 13. Conclusion

This project demonstrates how computer vision and deep learning can be used to solve real-world land classification problems using satellite imagery. It combines theoretical concepts with practical implementation.


Just tell 👍
