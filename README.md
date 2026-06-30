# Emotion and Autism Detection by Facial Recognition

A real-time facial analysis system that combines **Emotion Recognition** and **Autism Classification** using Deep Learning and Computer Vision. The application supports both **Laptop Webcam** and **ESP32-CAM** for live video streaming.

> **Note:** This project was developed as an academic/research prototype to demonstrate computer vision and deep learning concepts. It is **not** intended to be used as a medical diagnostic tool.

---

## 📌 Features

- Real-time face detection using OpenCV Haar Cascade
- Emotion recognition using Convolutional Neural Networks (CNN)
- Autism image classification using CNN
- Supports both Laptop Webcam and ESP32-CAM
- Live prediction with confidence scores
- Real-time visualization with bounding boxes and prediction labels

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 📁 Project Structure

```text
Emotion-and-Autism-Detection/
│
├── src/
│   ├── autexpression.py
│   └── combined.py
│
├── notebooks/
│   ├── TrainModel1.ipynb
│   └── TrainModel(Autism).ipynb
│
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

### Emotion Dataset

The emotion recognition model is trained using facial images belonging to the following classes:

- Angry
- Contempt
- Disgust
- Fear
- Happy
- Sadness
- Surprise

### Autism Dataset

Binary image classification:

- Autistic
- Non-Autistic

---

## 🧠 Deep Learning Models

### Emotion Recognition Model

- Input Size: **96 × 96 (Grayscale)**
- CNN Architecture
- Optimizer: **Adam**
- Loss Function: **Categorical Crossentropy**

Output Classes:

- Angry
- Contempt
- Disgust
- Fear
- Happy
- Sadness
- Surprise

---

### Autism Classification Model

- Input Size: **128 × 128 (RGB)**
- CNN Binary Classifier
- Optimizer: **Adam**
- Loss Function: **Binary Crossentropy**

Output Classes:

- Autistic
- Non-Autistic

---

## 🎥 Face Detection

The system uses OpenCV's built-in Haar Cascade classifier for detecting frontal human faces in real time.

```text
haarcascade_frontalface_default.xml
```

---

## ⚙️ System Workflow

```text
Laptop Webcam / ESP32-CAM
            │
            ▼
     Live Video Stream
            │
            ▼
     Face Detection
     (Haar Cascade)
            │
            ▼
      Face Extraction
      ┌───────────────┐
      │               │
      ▼               ▼
Emotion CNN      Autism CNN
      │               │
      └───────┬───────┘
              ▼
      Display Results
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/mohammedsiraj907/Emotion-and-Autism-Detection.git
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Laptop Webcam

Run:

```bash
python src/autexpression.py
```

### ESP32-CAM

1. Update the ESP32-CAM IP address inside:

```text
src/combined.py
```

2. Run:

```bash
python src/combined.py
```

---

## 📂 Trained Models

The trained `.keras` model files are **not included** in this repository because of GitHub's browser upload size limitations.

To recreate the models:

- Run **TrainModel1.ipynb** to generate the Emotion Recognition model.
- Run **TrainModel(Autism).ipynb** to generate the Autism Classification model.

---

## 📈 Future Improvements

- Improve autism dataset quality and diversity
- Replace Haar Cascade with modern face detection models
- TensorFlow Lite deployment
- Mobile application support
- Cloud deployment
- Multi-face tracking
- Performance optimization

---

## ⚠️ Disclaimer

This project was developed **for educational and research purposes only**.

The autism classification component is an experimental image classification model trained on the available dataset. It is **not clinically validated** and **must not be used** to diagnose Autism Spectrum Disorder (ASD) or make healthcare decisions.

---

## 👨‍💻 About

This repository presents a real-time facial analysis system built using **Python**, **TensorFlow**, **Keras**, and **OpenCV**. The system integrates facial emotion recognition and autism image classification through Convolutional Neural Networks (CNNs) and supports both laptop webcams and ESP32-CAM video streams.
