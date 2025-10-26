
# 🎨 ColorCortex

## A Machine Learning Pipeline for Color Classification


ColorCortex is an intelligent computer vision application built in Python that automatically identifies and classifies the colors of objects within a live video stream.
At its core is a **K-Nearest Neighbors (KNN)** machine learning model that acts as the "**cortex**," processing visual information to make smart, real-time predictions.

This project demonstrates a complete machine learning workflow — from **synthetic data generation** → **model training** → **live inference** — resulting in highly accurate color recognition in dynamic environments.

---

## ✅ Features

✔ Real-time color classification using a webcam
✔ Trained KNN model with 3,000+ synthetic samples
✔ Full automation — zero manual calibration
✔ Stable "detect object → classify color" pipeline
✔ Clean unified bounding boxes around detected objects

---

## 🛠️ Technologies

| Purpose                   | Technology    |
| ------------------------- | ------------- |
| Machine Learning          | Scikit-learn  |
| Real-time Computer Vision | OpenCV        |
| GUI                       | Tkinter       |
| Data Processing           | NumPy, Pandas |
| Model Persistence         | Joblib        |
| Image Format Conversion   | Pillow        |
| Language                  | Python 3      |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Mohitmhatre32/Colorcortex.git
cd colorcortex
```

### 2️⃣ Create and Activate a Virtual Environment

**Windows**

```bash
python -m venv venv
```
```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

This project runs in a **three-step pipeline**:

---

### ✅ Step 1: Generate Synthetic Dataset

Generates thousands of HSV color samples → `colors.csv`

```bash
python generate_dataset.py
```

---

### ✅ Step 2: Train the KNN Model

Trains the classifier and exports `color_model.pkl`

```bash
python train_model.py
```

---

### ✅ Step 3: Launch the Live Application

Run the real-time color classification UI

```bash
python app.py
```

📌 Point your camera at objects — predictions appear instantly with a label & bounding box!

---

## 🔬 Under the Hood — How It Works

1️⃣ **Frame Preprocessing**
→ Resize + blur to reduce noise

2️⃣ **Object Segmentation**
→ Edge detection / thresholding to detect shapes

3️⃣ **Color Extraction**
→ Compute average HSV per detected object

4️⃣ **AI Prediction**
→ Feed HSV to trained KNN to classify color

5️⃣ **Visualization**
→ Draw bounding box + label on original frame

A fast, clean, and stable detection pipeline ✅

---

## 📝 License

Distributed under the **MIT License**.
See [`LICENSE`](LICENSE) file for more information.

---


