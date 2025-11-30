# Heart Disease Prediction Using Traditional Machine Learning, Deep Neural Networks, and Transformer-Based Models

This repository contains the implementation for a graduate-level project in an **AI using Python** course.  
The goal of the project is to **predict heart disease** from structured health data using a **hybrid AI approach** that combines:

- Traditional Machine Learning
- Deep Neural Networks
- Transformer-based models for tabular data

The project compares **six models**:

- **XGBoost**
- **LightGBM**
- **Support Vector Classifier (SVC)**
- **Multi-Layer Perceptron (MLP)**
- **Convolutional Neural Network (1D CNN)**
- **TabTransformer**

---

## 🚀 Project Overview

This project tackles **binary classification**:  
> Predict whether a patient **has heart disease (1)** or **does not (0)**.

Key aspects:

- **Data preprocessing** for mixed numerical and categorical features
- **Handling missing values** (e.g., Alcohol Consumption)
- **Encoding categorical variables** (Label Encoding, One-Hot Encoding)
- **Feature scaling** using Min-Max normalization
- **Class imbalance handling** using **SMOTE (Synthetic Minority Over-sampling Technique)**
- Training and evaluating **six different models**
- Comparing models using:
  - Accuracy  
  - F1-score (macro)  
  - ROC-AUC  
  - Confusion matrices  
  - (Optionally) k-fold cross-validation

---

## 📂 Repository Structure

```text
.
├── heart-disease-classification-ml-dl-transformers.ipynb       # Main project script / notebook export
├── heart_disease.csv          # Dataset used in the project (tabular health data - from kaggle)
└── README.md                  # Project documentation (this file)

## 📝 Notes

- The notebook processes ~10,000 rows of data
- Training time varies by model (expect 10-30 minutes total on CPU, faster on GPU)
- Results may vary slightly due to random seeds in train-test splits
- Used Google Colab for this project. Please ensure that the file paths in the code match your Colab environment for a smooth run.
