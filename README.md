# Battery SOH and RUL Prediction

## Overview
This project focuses on predicting the State of Health (SOH) and Remaining Useful Life (RUL) of lithium-ion batteries using machine learning models. The objective is to analyze battery degradation patterns and evaluate model performance.

---

## Features
- Visualization of battery degradation (SOH vs cycle)
- Linear Regression and Random Forest models
- Model comparison using R², MSE, and MAE
- Feature importance analysis
- Battery health classification
- Basic fault detection logic

---

## Results and Insights
- Linear Regression performed better than Random Forest on this dataset
- Indicates strong linear relationships in battery degradation data
- Battery capacity (BCt) was identified as the most important feature influencing SOH
- RUL prediction achieved good accuracy (R² ≈ 0.93)

---

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
