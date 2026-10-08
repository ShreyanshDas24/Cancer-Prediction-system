# 🔬 Cancer Prediction System — Comprehensive Project Guide

> **Educational & Technical Guide for Machine Learning in Tabular Oncology Cytology**  
> *Target Convention: `0 = Malignant` (Positive Class of Interest) | `1 = Benign`*  
> *Stack: Python 3.11, scikit-learn, Streamlit, Plotly, NumPy, Pandas*

---

## Table of Contents
1. [Project Overview & Learning Objectives](#1-project-overview--learning-objectives)
2. [Dataset Origin & Clinical Background](#2-dataset-origin--clinical-background)
3. [Exploratory Data Analysis (EDA) & Multicollinearity](#3-exploratory-data-analysis-eda--multicollinearity)
4. [Data Preprocessing, Scaling & Leakage Prevention](#4-data-preprocessing-scaling--leakage-prevention)
5. [Feature Selection: 30 Baseline vs. 15 Production Features](#5-feature-selection-30-baseline-vs-15-production-features)
6. [Machine Learning Models & Mathematical Foundations](#6-machine-learning-models--mathematical-foundations)
   - [Logistic Regression](#61-logistic-regression)
   - [K-Nearest Neighbors (KNN)](#62-k-nearest-neighbors-knn)
   - [Decision Tree (CART)](#63-decision-tree-cart)
   - [Random Forest Classifier](#64-random-forest-classifier)
   - [Support Vector Machine (SVM)](#65-support-vector-machine-svm)
   - [Gradient Boosting Classifier](#66-gradient-boosting-classifier)
   - [Multi-Layer Perceptron (Neural Network)](#67-multi-layer-perceptron-neural-network)
7. [Evaluation Metrics & The Asymmetric Cost of Medical Errors](#7-evaluation-metrics--the-asymmetric-cost-of-medical-errors)
8. [Transparent Model Selection Methodology](#8-transparent-model-selection-methodology)
9. [Inference Pipeline & Risk Categorization](#9-inference-pipeline--risk-categorization)
10. [Streamlit UI Design & Visual Diagnostics](#10-streamlit-ui-design--visual-diagnostics)
11. [Codebase Architecture & Key Functions](#11-codebase-architecture--key-functions)
12. [Data Science Interview Defense & FAQ](#12-data-science-interview-defense--faq)

---

## 1. Project Overview & Learning Objectives

This project builds a clean, reproducible, end-to-end classification system for predicting breast tumor malignancy based on digitized nuclear measurements. 

### Why this project was refactored:
A prior legacy version of this repository contained a static web frontend claiming to run an advanced Support Vector Machine (RBF kernel, $C=100$) and Neural Network reaching 97.37% accuracy. In reality, the client-side JavaScript executed an arbitrary hardcoded linear combination (`-0.5, 0.8, -0.3...`) with a synthetic 1.5-second `setTimeout` delay to fake model inference.

This repository completely replaces that mocked structure with an **authentic, statistically validated, leak-free scikit-learn and Streamlit architecture**.

### Primary Objectives:
- Implement a leak-free data pipeline utilizing `sklearn.pipeline.Pipeline`.
- Compare 7 foundational machine learning algorithms on identical train/test splits.
- Contrast a 30-feature baseline against a 15-feature production pipeline derived via ANOVA F-statistic feature selection.
- Enforce explicit label handling: in the Wisconsin dataset, **0 represents Malignant and 1 represents Benign**.
- Provide transparent evaluation emphasizing Malignant Recall (Sensitivity) alongside ROC-AUC.

---

## 2. Dataset Origin & Clinical Background

### The Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
- **Collected by**: Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin, 1995).
- **Collection Technique**: Fine Needle Aspiration (FNA) of breast masses. A fine needle extracts fluid and cells, which are mounted on a slide, stained, and photographed under a microscope.
- **Image Processing**: Digital image analysis software (Xcyt) isolates cell nuclei contours and computes 10 physical characteristics:
  1. **Radius**: Mean of distances from center to perimeter points.
  2. **Texture**: Standard deviation of gray-scale values.
  3. **Perimeter**: Total distance around nuclear contour ($P \approx 2\pi r$).
  4. **Area**: Nuclear cross-sectional surface area ($A \approx \pi r^2$).
  5. **Smoothness**: Local variation in radius lengths.
  6. **Compactness**: $\frac{\text{Perimeter}^2}{\text{Area}} - 1.0$.
  7. **Concavity**: Severity of concave indentations in the contour.
  8. **Concave Points**: Number of distinct concave portions along the contour.
  9. **Symmetry**: Nuclear symmetry across major/minor axes.
  10. **Fractal Dimension**: "Coastline approximation" $- 1.0$.

For each of the 10 characteristics, 3 summary statistics were computed:
- **Mean** (features 1–10)
- **Standard Error (SE)** (features 11–20)
- **Worst / Extreme** (mean of the 3 largest measurements, features 21–30)
- **Total Features**: $10 \times 3 = 30$ continuous attributes.
- **Total Samples**: $N = 569$ biopsy instances.
- **Class Breakdown**: $212$ Malignant ($37.26\%$), $357$ Benign ($62.74\%$).

---

## 3. Exploratory Data Analysis (EDA) & Multicollinearity

### Simple Explanation
In geometry, if a circle gets bigger, its radius, its perimeter, and its area all grow together. Measuring radius, perimeter, and area on a cell nucleus means you are measuring the exact same physical property three times using different formulas.

### Technical & Mathematical Analysis
Multicollinearity occurs when two or more independent features in a dataset are strongly linearly correlated.

In our empirical analysis of the WDBC dataset, **21 feature pairs have a Pearson correlation $|r| > 0.90$**:
- $\text{mean radius} \leftrightarrow \text{mean perimeter}$: $r = 0.998$
- $\text{mean radius} \leftrightarrow \text{mean area}$: $r = 0.987$
- $\text{mean radius} \leftrightarrow \text{worst radius}$: $r = 0.970$
- $\text{mean perimeter} \leftrightarrow \text{worst perimeter}$: $r = 0.970$

#### Impact of Severe Multicollinearity:
In linear models (like Logistic Regression), the Ordinary Least Squares or maximum likelihood parameter variance is proportional to:
$$\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{(1 - R_j^2) \sum (x_{ij} - \bar{x}_j)^2}$$
where $\frac{1}{1 - R_j^2}$ is the **Variance Inflation Factor (VIF)**. When $R_j^2 \to 1.0$, $\text{Var}(\hat{\beta}_j) \to \infty$. This makes individual feature coefficients unstable, erratic, and difficult to interpret.

---

## 4. Data Preprocessing, Scaling & Leakage Prevention

### Simple Explanation
Imagine taking a test, but before the exam, the teacher accidentally lets you see the answer key. You would score higher, but the test wouldn't reflect your actual knowledge. 

In machine learning, **Data Leakage** happens when information from your test set leaks into the training process before the model is tested. If you calculate the average of all your data (including test data) to scale your features, your model has already "seen" the test set.

### Technical Implementation

#### 1. Standardization (Z-Score Normalization)
Many classifiers calculate Euclidean distance or use gradient descent:
$$z = \frac{x - \mu}{\sigma}$$
- $\mu$: Feature sample mean
- $\sigma$: Feature sample standard deviation

Without scaling:
- `mean area` ranges from $143.5$ to $2501.0$ $\text{mm}^2$ (span $> 2300$).
- `mean smoothness` ranges from $0.053$ to $0.163$ (span $\approx 0.11$).
The distance metric $\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$ would be dominated by area by a factor of $20,000$, rendering shape features completely invisible.

#### 2. Leak-Free Pipeline Encapsulation
In [src/preprocessing.py](file:///e:/Projects/Cancer-Prediction-System/src/preprocessing.py), scaling is wrapped inside scikit-learn's `Pipeline`:
```python
Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', estimator)
])
```
When performing Stratified 5-Fold Cross-Validation, `StandardScaler.fit()` is called **only on the 4 training folds**. The 1 validation fold is transformed using the parameters $(\mu_{\text{train}}, \sigma_{\text{train}})$ learned exclusively from the training folds.

---

## 5. Feature Selection: 30 Baseline vs. 15 Production Features

### The Two-Pipeline Design
To avoid forcing a clinician or user to input 30 separate continuous numbers while ensuring zero data leakage, we design two distinct pipelines:

1. **Experimental Baseline Pipeline (30 Features)**:
   - Evaluates all 30 measurements.
   - Demonstrates the theoretical upper performance bound with full feature dimensionality.
2. **Production Pipeline (15 Features)**:
   - Selects the top 15 features using ANOVA F-statistic strictly on `X_train`:
     $$F = \frac{\text{Between-group variance}}{\text{Within-group variance}} = \frac{\text{MSB}}{\text{MSW}}$$
   - The exact 15 feature names are extracted and saved to `models/feature_metadata.json`.
   - The production pipeline is fitted on `X_train[selected_15]` and expects **exactly those 15 features** at inference time.

### Selected 15 Production Features:
1. `mean radius` (Size)
2. `mean perimeter` (Size)
3. `mean area` (Size)
4. `mean compactness` (Shape)
5. `mean concavity` (Shape)
6. `mean concave points` (Shape)
7. `radius error` (Variability)
8. `perimeter error` (Variability)
9. `area error` (Variability)
10. `worst radius` (Size)
11. `worst perimeter` (Size)
12. `worst area` (Size)
13. `worst compactness` (Shape)
14. `worst concavity` (Shape)
15. `worst concave points` (Shape)

### Empirical Verification:
On the holdout test set ($N=114$):
- **Logistic Regression**: 30-feature AUC = `0.9954` vs. 15-feature AUC = `0.9931` (difference: $0.0023$)
- **Support Vector Machine**: 30-feature AUC = `0.9977` vs. 15-feature AUC = `0.9931` (difference: $0.0046$)
- **Neural Network**: 30-feature AUC = `0.9944` vs. 15-feature AUC = `0.9937` (difference: $0.0007$)

**Conclusion**: Retaining half the features preserves over **$99.5\%$ of discriminative power** while eliminating redundant collinear inputs and halving user data entry burden.

---

## 6. Machine Learning Models & Mathematical Foundations

### 6.1 Logistic Regression
- **Simple Intuition**: Draws a linear boundary. Calculates the log-odds of being malignant and passes it through an S-shaped curve (sigmoid) to output a probability between 0 and 1.
- **Mathematical Formulation**:
  $$P(Y = 0 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
  Loss function with L2 regularization:
  $$J(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
- **Assumptions**: Linear relationship between features and log-odds; absence of severe multicollinearity.

### 6.2 K-Nearest Neighbors (KNN)
- **Simple Intuition**: To diagnose a new biopsy, look at the $k$ most geometrically similar biopsies in the historical database and let them vote.
- **Mathematical Formulation**:
  Euclidean distance metric between patient $\mathbf{x}$ and stored sample $\mathbf{x}_i$:
  $$d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^p (x_j - x_{i,j})^2}$$
  Predicted class by plurality vote among the $k$ nearest instances $\mathcal{N}_k(\mathbf{x})$.
- **Assumptions**: Proximity in standardized feature space implies class similarity; highly sensitive to irrelevant features and scaling.

### 6.3 Decision Tree (CART)
- **Simple Intuition**: A flowchart of binary questions (e.g. *Is worst perimeter > 115 mm?*). Splits samples until terminal leaf nodes become mostly pure.
- **Mathematical Formulation**:
  Evaluates splits using Gini Impurity:
  $$I_G(t) = 1 - \sum_{i=0}^1 p(i \mid t)^2$$
  Split criterion selects the threshold maximizing Information Gain $\Delta I_G = I_G(parent) - \sum \frac{N_k}{N} I_G(child_k)$.
- **Assumptions**: Non-parametric; orthogonal linear decision boundaries parallel to feature axes. High variance if unpruned.

### 6.4 Random Forest Classifier
- **Simple Intuition**: Trains a committee of hundreds of diverse decision trees, each trained on a random sample of patients and a random subset of features, then averages their votes.
- **Mathematical Formulation**:
  - **Bootstrap Aggregating (Bagging)**: Generates $B$ bootstrap samples $\mathcal{D}_b$ drawn with replacement from $\mathcal{D}$.
  - **Feature Subspacing**: At each split, considers only $m = \sqrt{p}$ randomly selected features.
  $$\hat{P}(\text{Malignant} \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
- **Benefits**: Substantially reduces variance without increasing bias; provides Gini-based feature importances.

### 6.5 Support Vector Machine (SVM)
- **Simple Intuition**: Finds the decision boundary that leaves the widest possible empty "safety margin" between malignant and benign points. If data cannot be separated by a straight line, it bends the space using an RBF kernel.
- **Mathematical Formulation**:
  Maximum margin optimization:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \; \xi_i \ge 0$$
  Radial Basis Function (RBF) Kernel:
  $$K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$$
  Calibrated probabilities are obtained via **Platt Scaling**: fitting a logistic regression on the SVM decision values $f(\mathbf{x})$.

### 6.6 Gradient Boosting Classifier
- **Simple Intuition**: Trains trees sequentially. Each new tree focuses specifically on correcting the residual errors made by the previous trees.
- **Mathematical Formulation**:
  Given loss function $L(y, F(\mathbf{x})) = - [y \log p + (1-y) \log(1-p)]$, at step $m$:
  1. Compute pseudo-residuals (negative gradients):
     $$r_{im} = -\left[ \frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F=F_{m-1}}$$
  2. Fit a regression tree $h_m(\mathbf{x})$ to residuals $r_{im}$.
  3. Update model with learning rate shrinkage $\nu$:
     $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot \gamma_m h_m(\mathbf{x})$$

### 6.7 Multi-Layer Perceptron (Neural Network)
- **Simple Intuition**: Passes patient measurements through layers of interconnected mathematical neurons. Each neuron computes a weighted sum, applies a non-linear threshold (ReLU), and passes the signal to the output layer.
- **Mathematical Formulation**:
  For layer $l$:
  $$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
  $$\mathbf{a}^{(l)} = \text{ReLU}(\mathbf{z}^{(l)}) = \max(0, \mathbf{z}^{(l)})$$
  Output layer uses softmax / sigmoid activation. Trained via the Adam optimizer computing parameter updates via the chain rule (backpropagation):
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}} (\mathbf{a}^{(l-1)})^T$$

---

## 7. Evaluation Metrics & The Asymmetric Cost of Medical Errors

### The Confusion Matrix:
In our implementation, labels are strictly mapped as:
- `0 = Malignant` (Positive class)
- `1 = Benign` (Negative class)

| | Predicted Malignant (0) | Predicted Benign (1) |
| :--- | :--- | :--- |
| **True Malignant (0)** | **True Positive (TP)**: Cancer correctly identified | **False Negative (FN)**: Cancer missed (Critical!) |
| **True Benign (1)** | **False Positive (FP)**: False alarm, biopsy recommended | **True Negative (TN)**: Healthy correctly verified |

### Asymmetric Cost Matrix
In medical diagnostics:
- **Cost of FP**: Patient undergoes emotional stress and a follow-up core biopsy. Patient survives.
- **Cost of FN**: Malignant tumor goes untreated, progresses, and metastasizes. Potential patient fatality.
$$\text{Cost}(\text{FN}) \gg \text{Cost}(\text{FP})$$

### Metric Definitions:
1. **Malignant Recall (Sensitivity)**:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   Measures what fraction of true cancer patients were successfully detected.
2. **Benign Specificity**:
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
   Measures what fraction of healthy patients were spared unnecessary alarms.
3. **Malignant Precision (PPV)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   When the model flags a case as malignant, how often is it correct?
4. **F1-Score**:
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**:
   Calculates the probability that the model ranks a randomly chosen malignant sample higher than a randomly chosen benign sample across all possible decision thresholds $\tau \in [0, 1]$.

---

## 8. Transparent Model Selection Methodology

Rather than selecting a champion using holdout test data (which causes test-set leakage), our champion model is selected via a **strict, leak-free, safety-oriented ranking protocol on the training set**:

1. **Phase 1: Hyperparameter Tuning on $X_{\text{train}}$**: For each algorithm, hyperparameters are tuned using Stratified 5-Fold Cross-Validation strictly on the training set ($N=455$).
2. **Phase 2: Out-of-Fold CV Evaluation on $X_{\text{train}}$**: Predictions and probabilities are obtained out-of-fold across the 5 CV folds. Zero test set access.
3. **Phase 3: Champion Selection**: The champion model is chosen based exclusively on $X_{\text{train}}$ cross-validation performance, prioritizing Malignant Recall (Sensitivity) alongside ROC-AUC and F1:
   $$\text{Score}_{\text{CV}} = 0.40 \times \text{AUC}_{\text{CV}} + 0.35 \times \text{Recall}_{\text{CV}} + 0.25 \times \text{F1}_{\text{CV}}$$
4. **Phase 4: Single Holdout Test Evaluation**: Only after the champion is finalized is the holdout test set ($N=114$) evaluated **once** for unbiased reporting.

### Training Set Cross-Validation Results ($X_{\text{train}}$, $N=455$):

| Model | 15-Feat CV ROC-AUC | 15-Feat CV Malignant Recall | 15-Feat CV Accuracy | 15-Feat CV Malignant F1 | CV Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Neural Network (MLP)** 🏆 | **0.9925** | **0.9412** | **0.9516** | **0.9357** | **1 (Champion)** |
| **Logistic Regression** | 0.9918 | 0.9235 | 0.9451 | 0.9262 | 2 |
| **Support Vector Machine** | 0.9904 | 0.9000 | 0.9319 | 0.9080 | 3 |
| **Random Forest** | 0.9857 | 0.9176 | 0.9451 | 0.9258 | 4 |
| **Gradient Boosting** | 0.9832 | 0.9000 | 0.9341 | 0.9107 | 5 |
| **K-Nearest Neighbors** | 0.9816 | 0.9118 | 0.9451 | 0.9254 | 6 |
| **Decision Tree** | 0.9541 | 0.8765 | 0.9275 | 0.9003 | 7 |

*Champion Selected on $X_{\text{train}}$*: **Neural Network (MLP)** won Rank 1 based strictly on cross-validation on the training set, with the holdout test set completely untouched.

### Final Generalization on Untouched Holdout Test Set ($N=114$):

| Model | 15-Feat Test ROC-AUC | 15-Feat Test Recall (Malignant) | 15-Feat Test Accuracy | 15-Feat Test Precision | 15-Feat Test F1 | Test False Negatives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Neural Network (MLP)** 🏆 | **0.9937** | **0.9524** (40/42) | **0.9737** | 0.9756 | **0.9639** | **2** |
| **Support Vector Machine** | 0.9931 | 0.9286 (39/42) | 0.9561 | 0.9512 | 0.9398 | 3 |
| **Logistic Regression** | 0.9931 | 0.9286 (39/42) | 0.9561 | 0.9512 | 0.9398 | 3 |
| **Random Forest** | 0.9904 | 0.9286 (39/42) | 0.9561 | 0.9512 | 0.9398 | 3 |
| **Gradient Boosting** | 0.9921 | 0.9048 (38/42) | 0.9474 | 0.9500 | 0.9268 | 4 |
| **K-Nearest Neighbors** | 0.9800 | 0.9048 (38/42) | 0.9386 | 0.9268 | 0.9157 | 4 |
| **Decision Tree** | 0.9793 | 0.8810 (37/42) | 0.9474 | 0.9737 | 0.9250 | 5 |

---

## 9. Inference Pipeline & Risk Categorization

In [src/predict.py](file:///e:/Projects/Cancer-Prediction-System/src/predict.py), predictions avoid clinical diagnostic terms and instead output **statistical risk categories**:

| Predicted Malignant Probability $P(\text{Malignant})$ | Statistical Model Risk Category |
| :--- | :--- |
| $< 10\%$ | Very Low |
| $10\% - 35\%$ | Low |
| $35\% - 65\%$ | Intermediate / Ambiguous |
| $65\% - 85\%$ | Elevated |
| $\ge 85\%$ | High |

### Feature Deviation Analysis (Z-Scores vs. Benign Baseline):
For every prediction, the service computes how many standard deviations ($z$) the patient's measurements sit above or below the benign cohort mean:
$$z_i = \frac{x_i - \mu_{\text{benign}, i}}{\sigma_{\text{benign}, i}}$$
This provides transparent interpretability showing *why* the model made its prediction (e.g. `worst concavity` is $+3.2\sigma$ above benign average).

---

## 10. Streamlit UI Design & Visual Diagnostics

The Streamlit interface in [app.py](file:///e:/Projects/Cancer-Prediction-System/app.py) provides four comprehensive interactive tabs:

1. **Tab 1: Diagnostic Prediction**:
   - Organized into 3 clinical categories (Size, Shape, Variability).
   - One-click buttons to load **Typical Benign**, **Typical Malignant**, or **Borderline** patient presets.
   - Live probability bar, risk category badge, and interactive horizontal Plotly deviation chart.
2. **Tab 2: Model Comparison & Benchmarks**:
   - Interactive leaderboard table with highlighted maximums.
   - Side-by-side grouped bar chart comparing 15-feature vs 30-feature ROC-AUC.
   - Interactive ROC curves and confusion matrix heatmap.
   - Educational threshold sensitivity vs specificity trade-off curve.
3. **Tab 3: Exploratory Data Analysis (EDA)**:
   - Target class distribution pie chart.
   - Feature distribution box plots with individual sample jitter.
   - Interactive Pearson correlation heatmap demonstrating multicollinearity.
4. **Tab 4: Educational Guide**:
   - Quick architectural overview and interview prep notes.

---

## 11. Codebase Architecture & Key Functions

```
Cancer-Prediction-System/
├── app.py                      # Main Streamlit web application
├── train_model.py              # CLI training pipeline runner
├── requirements.txt            # Pinned Python 3.11 dependencies
├── src/
│   ├── data_loader.py          # Data validation & stratified splitting
│   ├── preprocessing.py        # Pipelines & leak-free feature selection
│   ├── train.py                # Hyperparameter tuning (GridSearchCV) & training
│   ├── evaluate.py             # Metrics, ROC curves, threshold trade-offs
│   └── predict.py              # Single-sample inference & risk tiers
├── models/
│   ├── best_model.joblib       # Champion pipeline (Scaler + MLP)
│   ├── all_models.joblib       # All 7 trained 15-feature candidate pipelines
│   ├── metrics_summary.json    # Verified evaluation metrics
│   └── feature_metadata.json   # 15 feature stats & descriptions
└── tests/
    ├── test_data_loader.py     # Data integrity & shape tests
    ├── test_preprocessing.py   # Leak-free transformation tests
    └── test_predict.py         # Prediction & validation tests
```

---

## 12. Data Science Interview Defense & FAQ

### Q1: "How did you prevent data leakage during feature selection?"
> *"In many student projects, feature selection algorithms like `SelectKBest` or `RFE` are run on the full dataset before splitting. That leaks target distribution information from the test set into training. In our pipeline, `select_features_on_training_data` is fitted **exclusively on the 80% training split**. The 15 selected feature names are stored as a fixed schema, and the test set is transformed strictly during final evaluation."*

### Q2: "Why is accuracy misleading for this problem?"
> *"The WDBC dataset has an imbalanced class distribution ($62.7\%$ benign, $37.3\%$ malignant). A naive 'dummy' classifier predicting benign for every patient would achieve $62.7\%$ accuracy while having a $0\%$ malignant recall, resulting in fatal medical consequences. In oncology, **Recall on the malignant class (Sensitivity)** and **ROC-AUC** are the primary safety-critical metrics."*

### Q3: "What is the label convention in scikit-learn's WDBC dataset?"
> *"In scikit-learn's `load_breast_cancer`, label `0` is Malignant and label `1` is Benign. If an engineer computes `recall_score(y_true, y_pred)` without passing `pos_label=0`, scikit-learn defaults to `pos_label=1`, reporting **Benign Recall instead of Malignant Recall**. Our codebase explicitly enforces `pos_label=0` across all metric calculations."*

### Q4: "Why did you choose a 15-feature pipeline over all 30 features?"
> *"The 30 features suffer from severe geometric multicollinearity—21 pairs have $|r| > 0.90$. Furthermore, asking a doctor to manually input 30 continuous numbers in a UI introduces high human error and friction. Our empirical experiments demonstrated that reducing to 15 features via ANOVA F-statistics preserved over $99.5\%$ of the ROC-AUC ($0.9937$ vs $0.9944$) while stabilizing parameter variance and creating a practical user interface."*
