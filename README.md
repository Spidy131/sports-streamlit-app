# sports-streamlit-app
100-Class Sports Image Classifier using Transfer Learning (MobileNetV2) with Fine-Tuning and Streamlit Web Deployment | Achieved 90% Test Accuracy
🏆 Sports Image Classifier – 100 Classes
app link:https://sports-app-app-npelhy2snvkbjwgk5rb6lf.streamlit.app/
This project is a deep learning-based multi-class image classification system that identifies 100 different sports categories using Transfer Learning (MobileNetV2) and fine-tuning techniques.

The model achieves ~90% test accuracy on 500 unseen test images.

🚀 Project Highlights

🔹 100-class multi-class classification

🔹 13,492 training images

🔹 Transfer Learning using MobileNetV2

🔹 Data Augmentation applied

🔹 Fine-Tuning of last convolution layers

🔹 Achieved ~90% test accuracy

🔹 Confusion Matrix & Classification Report generated

🔹 Deployed using Streamlit Web App

🧠 Model Architecture

Base Model: MobileNetV2 (Pre-trained on ImageNet)

Image Size: 224 × 224

Optimizer: Adam

Fine-Tuning Learning Rate: 1e-5

Output Layer: Softmax (100 Classes)

📊 Performance Metrics
Metric	Value
Test Accuracy	~90%
Macro F1-Score	~0.89
Weighted F1-Score	~0.89
Total Classes	100
Total Training Images	13,492

Evaluation performed using:

Confusion Matrix

Precision

Recall

F1-Score

🖥 Streamlit Web Application

The model is deployed using Streamlit.

Features:

Upload image

Real-time prediction

Confidence score display

Clean interactive UI

Project Structure:
sports-streamlit-app/
│
├── app.py
├── sports_100_model_finetuned.h5
├── requirements.txt
└── README.md

Installation

Clone the repository:
git clone https://github.com/your-username/sports-streamlit-app.git
cd sports-streamlit-app
Install dependencies:
pip install -r requirements.txt
Run Streamlit app:
streamlit run app.py

#Technologies Used

Python

TensorFlow / Keras

MobileNetV2

Streamlit

NumPy

PIL

Scikit-learn
