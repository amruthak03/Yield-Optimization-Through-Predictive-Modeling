# Yield-Optimization-Through-Predictive-Modeling

This project focuses on yield optimization using predictive modeling techniques to identify factors that impact product quality in semiconductor manufacturing. The dataset used is the UCI SECOM dataset, which contains measurements from semiconductor production and a binary outcome: Pass or Fail. This repository contains the code to preprocess the data, apply different machine learning models, and evaluate their performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Evaluation](#model-evaluation)
- [Streamlit Interface](#streamlit-interface)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview
The main objective of this project is to predict the quality of products (Pass or Fail) using data collected from semiconductor manufacturing processes. The dataset includes various sensor measurements, which are analyzed using machine learning techniques to optimize yield and improve product quality.

### Key Steps:
1. Data cleaning and preprocessing (handling missing values, removing low-variance features, and collinearity).
2. Addressing class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
3. Training multiple machine learning models including Random Forest, Logistic Regression, Gradient Boosting, and SVM.
4. Evaluating models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
5. Creating an interactive Streamlit dashboard to visualize results and explore model predictions.

## Data Preprocessing
- Handling Missing Values: The dataset contains missing values that are imputed using the mean strategy.
- Variance Threshold: Low-variance features are removed to reduce noise in the model.
- Correlation Analysis: Highly collinear features are dropped to avoid multicollinearity.
- Class Imbalance: SMOTE is used to generate synthetic samples for the minority class to balance the target classes (Pass/Fail).

## Model Evaluation
The models are evaluated using the following metrics:

- Classification Report: Precision, Recall, F1-Score
- Confusion Matrix: Visualizes true positives, false positives, true negatives, and false negatives.
- ROC-AUC Score: Measures the area under the receiver operating characteristic curve to assess the tradeoff between true positive rate and false positive rate.

## Streamlit Interface
The project includes a Streamlit interface to interact with the data and models. The interface allows users to:

- Visualize target class distribution (Pass/Fail).
- Inspect feature importance and model evaluation metrics.
- Test the models with custom data inputs.

## Installation
To set up the environment and install the required dependencies, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/amruthak03/yield-optimization-through-predictive-modeling.git
    ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Preprocess the data: Run the main Python script to preprocess the data, remove low-variance and collinear features, and address class imbalance.
2. Train the models: Use the provided code to train and evaluate the models.
3. Visualize results: You can visualize results using the confusion matrix, ROC-AUC scores, and SHAP values for feature importance.
4. Run the Streamlit app: Launch the Streamlit app to interact with the results.

