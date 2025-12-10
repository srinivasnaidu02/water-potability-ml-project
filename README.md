# 💧 Water Potability Prediction (Machine Learning + Streamlit)

This project predicts whether water is **potable (safe for drinking)** based on its chemical properties.  
It includes **data preprocessing, Exploratory Data Analysis (EDA), model training, evaluation, and an interactive Streamlit web app** for making predictions.

---

## 📌 Project Overview

- Uses real-world water quality data to classify water as **potable** or **not potable**.
- Handles **missing values** and scales features for better model performance.
- Trains a **Random Forest Classifier** (or similar ML model) to make predictions.
- Exposes a **Streamlit web interface** where users can input water parameters and instantly see the prediction.

---

## 🧪 Dataset

- File: `water_potability.csv`
- Each row represents a water sample with chemical properties such as:
  - `ph`
  - `Hardness`
  - `Solids`
  - `Chloramines`
  - `Sulfate`
  - `Conductivity`
  - `Organic_carbon`
  - `Trihalomethanes`
  - `Turbidity`
  - `Potability` (Target: 0 = Not Potable, 1 = Potable)

> The dataset is commonly available on Kaggle as **"Water Potability"**.

---

## 🏗️ Project Structure

```bash
water-potability-ml-project/
│
├── app.py                 # Streamlit application
├── water_potability.csv   # Dataset
├── requirements.txt       # Python dependencies
├── model.pkl              # Trained ML model (optional, if saved)
└── README.md              # Project documentation
