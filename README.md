# Satellite-Image-Enhancement-Feature-Extraction-and-Land-Classification

Satellite Image Land Classification System
1. Overview

This project is a deep learning-based system that classifies satellite images into different land categories such as Forest, River, SeaLake, Residential, and Industrial. It integrates image preprocessing, enhancement, and classification into a single pipeline. A Streamlit-based interface allows users to upload images and get real-time predictions with confidence scores.

The system is designed to demonstrate practical implementation of computer vision concepts along with modern deep learning techniques.

2. Project Architecture
satellite_image_land_classification/
│
├── app.py                          # Streamlit application
├── model/
│   ├── land_classifier_model.keras
│   └── class_labels.json
│
├── utils/
│   ├── classification.py           # Prediction pipeline
│   ├── image_enhancement.py        # Enhancement pipeline
│   └── feature_extraction.py       # Feature extraction methods
│
├── data/
│   ├── sample_images/
│   └── outputs/
│
├── requirements.txt
└── README.md


3. Model Details (Important)
Model Type
Convolutional Neural Network (CNN)
Based on MobileNetV2 architecture (lightweight and efficient)
Input Specifications
Input Size: 224 × 224 × 3
Color Format: RGB
Normalization: Image pixel values scaled to [0, 1]
Output
Softmax layer for multi-class classification
Output shape: (1, 10)
Classes
AnnualCrop
Forest
HerbaceousVegetation
Highway
Industrial
Pasture
PermanentCrop
Residential
River
SeaLake
Training Details
Dataset: EuroSAT (satellite images)
Loss Function: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
Important Note (Why your model was giving wrong results)
If preprocessing during inference ≠ preprocessing during training → wrong predictions
Example issues:
Wrong normalization (0–255 instead of 0–1)
BGR vs RGB mismatch
Using enhanced image instead of raw image
Label order mismatch
4. Tech Stack
Programming Language
Python
Libraries Used
TensorFlow / Keras
OpenCV
NumPy
Streamlit
PIL
5. Installation (All pip commands)
pip install tensorflow
pip install opencv-python
pip install numpy
pip install streamlit
pip install pillow
pip install matplotlib
pip install scikit-learn

Or use single command:

pip install tensorflow opencv-python numpy streamlit pillow matplotlib scikit-learn
6. How to Run the Project
Method 1: Normal Command Line
streamlit run app.py
Method 2: Using .bat File (Windows)

Create a file named run_app.bat:

@echo off
cd /d %~dp0
call venv\Scripts\activate
streamlit run app.py
pause

Then double-click the file.

Method 3: Without Virtual Environment
python -m streamlit run app.py
Method 4: VS Code
Open project folder
Open terminal
Run:
streamlit run app.py
7. Workflow of the System
User uploads satellite image
Image is validated and preprocessed
Resized to model input size
Converted to RGB
Normalized
Passed into trained CNN model
Model outputs class probabilities
Top predictions displayed
8. Image Enhancement Module

This module improves image quality but should not be used for classification input.

Techniques Used:
CLAHE (contrast enhancement)
Noise removal (denoising)
Sharpening
Gamma correction
Histogram processing
9. Syllabus Topics Covered (Very Important)
1. Digital Image Formation and Low-Level Processing
Image representation using pixel matrices
Image transformations (scaling, resizing)
Filtering and convolution (denoise, blur)
Histogram processing and enhancement (CLAHE)
Image enhancement techniques
2. Depth Estimation and Multi-Camera Views
Conceptual understanding of perspective projection
Basic geometric transformations used in preprocessing
Not fully implemented but theoretical foundation used
3. Feature Extraction and Image Segmentation
Edge detection concepts (used in enhancement pipeline)
Texture understanding in satellite images
CNN automatically extracts features (deep feature extraction)
Concepts of HOG, SIFT, etc. replaced by deep learning
4. Pattern Analysis and Motion Analysis
Classification using supervised learning
CNN model (Artificial Neural Network)
Probability-based decision making (Softmax)
Accuracy evaluation
Confusion matrix and classification report
5. Shape from X
Light and surface interaction (important for satellite images)
Texture and color-based classification
Reflectance understanding (implicitly learned by CNN)
10. Common Issues and Fixes
Issue: Model predicts same class (SeaLake always)

Possible Reasons:

Wrong preprocessing
Label mismatch
Model overfitting
Using enhanced image

Fix:

Ensure RGB format
Normalize correctly
Use original image
Verify label order
Issue: Model not loading

Fix:

Convert .h5 → .keras
Match TensorFlow version
Use custom layer handling (TrueDivide)
11. Applications
Environmental monitoring
Urban planning
Agriculture analysis
Water body detection
Remote sensing
12. Future Improvements
Improve model accuracy
Add more datasets
Add Grad-CAM visualization
Deploy on cloud
Add real-time satellite data
13. Conclusion

This project successfully demonstrates the integration of computer vision and deep learning for real-world land classification problems. It combines theoretical concepts from image processing with practical implementation using modern AI techniques
