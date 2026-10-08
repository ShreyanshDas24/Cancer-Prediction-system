# 🔬 Advanced Cancer Prediction System

> **A Reproducible Machine Learning Study & Interactive Streamlit Application**  
> *Wisconsin Diagnostic Breast Cancer (WDBC) Dataset | Python 3.11 & scikit-learn*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)]()

---

## ⚠️ Educational & Research Disclaimer

**This project is developed exclusively for educational and scientific research purposes.**  
It calculates statistical probability estimates based on machine-learning models trained on the historical 1995 Wisconsin Diagnostic Breast Cancer dataset. **It does not provide medical diagnoses, clinical guidance, or treatment recommendations.** Never use this tool for healthcare or medical decisions.

---

## 🎯 Target Label Convention

In the scikit-learn Wisconsin Diagnostic Breast Cancer dataset:
- **`0 = Malignant`** (Safety-critical positive class of interest)
- **`1 = Benign`** (Negative class)

All evaluation metrics, confusion matrices, and risk categorizations in this repository explicitly enforce this convention. Malignant Recall (Sensitivity) measures the detection rate of actual malignant cases ($0$).

---

## 📊 Verified Model Benchmark Results

All metrics below were **reproduced deterministically** by training from scratch on the 80% training set ($N=455$) and evaluating on the unseen holdout test set ($N=114$):

| Model Paradigm | 15-Feat ROC-AUC | 15-Feat Malignant Recall | 15-Feat Accuracy | 15-Feat Malignant F1 | 30-Feat Baseline ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Neural Network (MLP)** 🏆 *(Champion)* | **0.9937** | **0.9524** (40/42) | **0.9737** | **0.9639** | 0.9944 |
| **Support Vector Machine (RBF)** | 0.9931 | 0.9286 (39/42) | 0.9561 | 0.9398 | **0.9977** |
| **Logistic Regression (L2)** | 0.9931 | 0.9286 (39/42) | 0.9561 | 0.9398 | 0.9954 |
| **Random Forest Classifier** | 0.9904 | 0.9286 (39/42) | 0.9561 | 0.9398 | 0.9934 |
| **Gradient Boosting Classifier** | 0.9921 | 0.9048 (38/42) | 0.9474 | 0.9268 | 0.9950 |
| **K-Nearest Neighbors (KNN)** | 0.9800 | 0.9048 (38/42) | 0.9386 | 0.9157 | 0.9917 |
| **Decision Tree (CART)** | 0.9793 | 0.8810 (37/42) | 0.9474 | 0.9250 | 0.9448 |

> **Champion Selection Rationale**: MLP was selected as the champion based on the project's predefined cross-validation selection criterion applied exclusively to the training data. It subsequently achieved the strongest holdout-test results, including 95.24% malignant recall, 0.9937 ROC-AUC, and 97.37% accuracy.

---

## 🏛️ System Architecture

```
User Input (15 Features in Streamlit UI)
                   ↓
      CancerPredictor (src/predict.py)
                   ↓
scikit-learn Pipeline (StandardScaler → Model)
                   ↓
P(Malignant) Probability & Risk Categorization
                   ↓
Feature Deviation Z-Scores (vs. Benign Cohort)
```

### Two-Pipeline Strategy:
1. **Experimental Baseline Pipeline (30 Features)**: Retains all original measurements to provide a reference point for evaluating the reduced 15-feature pipeline.
2. **Production Pipeline (15 Features)**: Selects 15 features using ANOVA F-statistic strictly on `X_train` without data leakage. The reduced feature set retains performance close to the 30-feature baseline while reducing the number of user inputs.

---

## 📁 Repository Structure

```
Cancer-Prediction-System/
├── app.py                          # Streamlit clinical dashboard application
├── train_model.py                  # CLI pipeline runner (trains & saves models)
├── requirements.txt                # Pinned Python dependencies
├── README.md                       # Project overview and reproduction guide
├── .gitignore                      # Git exclusion rules
│
├── data/
│   ├── raw/                        # Raw WDBC dataset cache
│   └── processed/                  # Deterministic train/test split CSVs
│
├── models/
│   ├── best_model.joblib           # Champion 15-feature production pipeline
│   ├── all_models.joblib           # All 7 trained 15-feature pipelines
│   ├── metrics_summary.json        # Verified test metrics & curves data
│   └── feature_metadata.json       # 15 feature stats, ranges, units, medians
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data acquisition, validation & splitting
│   ├── preprocessing.py            # Pipelines & leak-free feature selection
│   ├── train.py                    # 7 models, GridSearch, Stratified CV
│   ├── evaluate.py                 # Metrics with Malignant as positive class
│   └── predict.py                  # Inference service & risk categorization
│
├── assets/                         # Generated ROC curves, confusion matrices
│
├── docs/
│   ├── PROJECT_GUIDE.md            # Deep educational manual (math, theory, code)
│   └── DEVELOPMENT_LOG.md          # Chronological engineering journal
│
└── tests/
    ├── test_data_loader.py         # Data validation tests
    ├── test_preprocessing.py       # Pipeline & feature selection tests
    └── test_predict.py             # Inference & risk categorization tests
```

---

## 🚀 Quickstart & Reproduction Guide

### Prerequisites
- Python 3.11 (with `pip`)

### 1. Install Dependencies
```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. Train Models & Generate Benchmarks
Execute the end-to-end training pipeline. This performs leak-free feature selection, cross-validation, hyperparameter tuning, model artifact serialization, and metric reporting:
```bash
py -3.11 train_model.py
```

### 3. Launch the Streamlit Web Application
```bash
py -3.11 -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 4. Run Automated Test Suite
```bash
py -3.11 -m pytest tests/
```

---

## 📚 Deep-Dive Documentation

For detailed technical and educational walkthroughs, consult:
- **[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md)**: Deep mathematical derivations, model assumptions, clinical metric theory, and 10 common data science interview defense questions.
- **[`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)**: Engineering journal documenting legacy code audit, bug fixes, experiments, and lessons learned.
