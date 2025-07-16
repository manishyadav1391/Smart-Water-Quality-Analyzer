# 💧 AI-Based Water Quality Analyzer

> Predict whether a water sample is safe for drinking using Machine Learning  
> 📌 **Aligned with UN SDG 6 – Clean Water and Sanitation**

---

## 📌 Project Overview

This project aims to build a machine learning-based solution that predicts the **potability of water** (i.e., whether it's safe to drink) using physicochemical parameters such as pH, turbidity, sulfate, etc.

The model is wrapped into a clean and responsive **Streamlit web app** for real-time usage. This tool enables users and organizations to perform quick and accessible water quality checks, helping prevent waterborne diseases and supporting SDG 6.

---

## 🎯 Problem Statement

Traditional water testing methods are often costly and require laboratory infrastructure. This creates accessibility issues, especially in underserved areas.

The goal of this project is to:
- Develop an AI model that predicts water safety based on sample parameters.
- Deploy a web interface for users to interactively test water samples.
- Provide confidence scores and health tips based on model output.

---

## 📊 Dataset

- 📂 **Source**: [Kaggle – Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- 💡 **Features**:
  - `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`
  - `Conductivity`, `Organic Carbon`, `Trihalomethanes`, `Turbidity`
- 🎯 **Label**: `Potability` (1 = Safe, 0 = Unsafe)

Missing values are handled using mean imputation.

---

## 🧠 Machine Learning Model

- ✅ Algorithm: **Random Forest Classifier**
- 🔍 Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ⚙️ Tools: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

## 🖥 Web Application (Streamlit)

### 🧪 Features:
- Sliders for entering water quality metrics
- Predict button with safety result
- Confidence percentage
- Visualizations like correlation heatmap (optional)
- Health tips based on outcome

### 📷 Screenshot
![App Screenshot](assets/screenshot.png)

---

## 📦 Project Structure

