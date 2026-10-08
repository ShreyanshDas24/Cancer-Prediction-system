# 📓 Development Log & Engineering Journal

This document records the chronological development history, engineering decisions, empirical experiments, bugs diagnosed, fixes applied, and lessons learned during the construction of the Cancer Prediction System.

---

## Log Entry 1: Legacy Code Audit & Technical Debt Identification
- **Date**: 2026-10-09
- **Author**: Antigravity Engineering Team
- **Status**: Completed

### Context & Discovery:
A thorough inspection of the original repository (`ShreyanshDas24/Cancer-Prediction-System`) was conducted. The project was presented as an advanced machine learning platform achieving 97.37% accuracy using an RBF Support Vector Machine and Neural Network.

### Issues Diagnosed:
1. **Mocked Client-Side Inference**: In `app.js` (lines 37–42, 267–278), the "SVM prediction" was revealed to be a hardcoded 15-element linear dot product with arbitrary weights (`-0.5, 0.8, -0.3...`) passed into a logistic sigmoid function.
2. **Artificial Delay**: Line 213 in `app.js` included `await new Promise(resolve => setTimeout(resolve, 1500));` to simulate network processing.
3. **Missing Python Machine Learning Code**: No training scripts, notebooks, or serialized `.joblib` model files existed anywhere in the repository.
4. **Orphaned Assets**: `cancer_prediction_data.json`, `model_comparison_results.csv`, and chart PNGs were static files never read or referenced by the running application.

### Decision:
Completely redesign the application into an authentic, reproducible, modular Python machine learning architecture powered by **scikit-learn** and **Streamlit**.

---

## Log Entry 2: Dataset Investigation & Multicollinearity Analysis
- **Date**: 2026-10-09
- **Status**: Completed

### Investigation:
We inspected the Wisconsin Diagnostic Breast Cancer (WDBC) dataset using Python 3.11 (`sklearn.datasets.load_breast_cancer`). The dataset contains 569 instances and 30 continuous features.

### Empirical Findings:
1. **Target Distribution**:
   - Class `0`: Malignant ($N = 212$, $37.26\%$)
   - Class `1`: Benign ($N = 357$, $62.74\%$)
   *Crucial finding*: In scikit-learn, label `0` is Malignant and label `1` is Benign. Many student projects erroneously treat 1 as positive, accidentally reporting benign recall as malignant recall.
2. **Extreme Geometric Collinearity**:
   Correlation matrix analysis revealed **21 feature pairs with Pearson $|r| > 0.90$**:
   - `mean radius` $\leftrightarrow$ `mean perimeter`: $r = 0.998$
   - `mean radius` $\leftrightarrow$ `mean area`: $r = 0.987$
   - `worst radius` $\leftrightarrow$ `worst perimeter`: $r = 0.970$
   - `mean perimeter` $\leftrightarrow$ `worst area`: $r = 0.942$

### Lesson Learned:
Radius, perimeter, and area are deterministic geometric transformations ($P \approx 2\pi r$, $A \approx \pi r^2$). Feeding all 30 features into linear models causes variance inflation and unstable coefficients.

---

## Log Entry 3: Feature Selection & Two-Pipeline Architecture
- **Date**: 2026-10-09
- **Status**: Completed

### Problem Statement:
How do we balance the statistical completeness of all 30 features with the practical requirement of an interactive user interface that does not overwhelm the user with 30 manual numeric inputs?

### Architecture Decision:
Create two explicitly separated pipelines:
1. **Experimental Baseline Pipeline (30 Features)**: Evaluates all measurements as an academic benchmark.
2. **Production Pipeline (15 Features)**: Selects top 15 features using ANOVA F-statistic (`SelectKBest(score_func=f_classif, k=15)`).

### Strict Leak-Free Protocol:
- Feature selection is fitted **strictly on `X_train`** ($N=455$).
- The exact 15 selected feature names are persisted to `models/feature_metadata.json`.
- The production pipeline is fitted on `X_train[selected_15]` and expects **only those 15 features** at inference time. The Streamlit UI collects only those 15 features.

---

## Log Entry 4: Model Training, Hyperparameter Grid Search & Results
- **Date**: 2026-10-09
- **Status**: Completed

### Execution:
In `src/train.py`, we implemented Stratified 5-Fold Cross-Validation with `GridSearchCV` on the training set ($N=455$) and evaluated all 7 models on the unseen holdout test set ($N=114$, $42$ Malignant, $72$ Benign).

### Verified Test Benchmark Results:

| Model | 15-Feat ROC-AUC | 15-Feat Malignant Recall | 15-Feat Accuracy | 15-Feat Precision | 15-Feat F1 | False Negatives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Neural Network (MLP)** 🏆 | **0.9937** | **0.9524** (40/42) | **0.9737** | 0.9756 | **0.9639** | **2** |
| **Support Vector Machine** | 0.9931 | 0.9286 (39/42) | 0.9561 | 0.9512 | 0.9398 | 3 |
| **Logistic Regression** | 0.9931 | 0.9286 (39/42) | 0.9561 | 0.9512 | 0.9398 | 3 |
| **Random Forest** | 0.9904 | 0.9286 (39/42) | 0.9561 | 0.9512 | 0.9398 | 3 |
| **Gradient Boosting** | 0.9921 | 0.9048 (38/42) | 0.9474 | 0.9500 | 0.9268 | 4 |
| **K-Nearest Neighbors** | 0.9800 | 0.9048 (38/42) | 0.9386 | 0.9268 | 0.9157 | 4 |
| **Decision Tree** | 0.9793 | 0.8810 (37/42) | 0.9474 | 0.9737 | 0.9250 | 5 |

### Champion Model Selection Rationale:
In oncology screening, missing a malignancy (False Negative) is clinically the most critical failure mode.
**Neural Network (MLP)** was selected as the champion model because it achieved:
- Highest Malignant Recall ($95.24\%$, missing only 2 cases out of 42)
- Highest ROC-AUC ($0.9937$)
- Highest Test Accuracy ($97.37\%$, $111/114$ correct)

---

## Log Entry 5: Streamlit Application & Ethical Framing
- **Date**: 2026-10-09
- **Status**: Completed

### Implementation in `app.py`:
1. **Non-Clinical Framing**: Removed all clinical recommendation wording ("consult oncologist", "treatment options"). Substituted with neutral machine-learning terminology:
   - "predicted class"
   - "model probability"
   - "model risk category"
2. **Prominent Medical Disclaimer**: Placed at the top of the interface clarifying that the system is an educational statistical model.
3. **Interactive Presets**: Added one-click loaders for *Typical Benign*, *Typical Malignant*, and *Borderline* samples using median stats from `models/feature_metadata.json`.
4. **Interpretability**: Added Plotly horizontal bar chart displaying patient feature z-scores relative to the benign population.
5. **Multi-Model Benchmark & Threshold Analysis**: Embedded live ROC curves, confusion matrices, and educational sensitivity vs. specificity trade-off curves.

---

## Log Entry 6: Verification & Test Suite
- **Date**: 2026-10-09
- **Status**: Completed

### Unit Tests Implemented:
- `tests/test_data_loader.py`: Verifies dataset shapes, zero nulls, label mapping, and stratified ratio preservation.
- `tests/test_preprocessing.py`: Verifies leak-free feature selection, pipeline construction, and metadata calculation.
- `tests/test_predict.py`: Verifies single-sample prediction, probability calculation, missing feature validation, and non-numeric value rejection.

### Execution Results:
All 11 unit tests passed in 4.29 seconds via `py -3.11 -m pytest tests/`.
`app.py` successfully compiled with zero errors.

---

## Log Entry 7: Audit of Model-Selection Methodology & Test-Leakage Elimination
- **Date**: 2026-10-09
- **Status**: Completed

### Audit Finding:
A strict audit of `src/train.py` was conducted to verify test-set isolation. We verified:
1. **Hyperparameters**: Tuned exclusively on `X_train` via 5-fold cross-validation inside `GridSearchCV`. (Pass)
2. **Champion Selection**: The original implementation evaluated all candidate models on `X_test` inside the loop and passed `production_test_metrics` into `select_champion_model()`. This constituted a **methodological leakage**: the champion model was chosen based on holdout test set performance rather than training cross-validation. (Issue Identified)

### Protocol Fix Applied:
Refactored `src/train.py` and `train_model.py` to enforce the following strict sequence:
1. **Phase 1 (Training & CV)**: Hyperparameters are tuned on `X_train`. Out-of-fold cross-validation predictions and probabilities on `X_train` are generated using `cross_val_predict(cv=StratifiedKFold(n_splits=5))`. Full CV metrics (`cv_metrics`) are computed on `X_train`.
2. **Phase 2 (Champion Model Selection)**: `select_champion_model()` is executed **strictly on the out-of-fold CV metrics of `X_train`**. The test set is completely locked and inaccessible.
3. **Phase 3 (Single Test Evaluation)**: Only after the champion has been finalized is the holdout test set (`X_test`, $N=114$) evaluated **once** to report final generalization metrics.

### Verification:
Under strict `X_train` out-of-fold cross-validation:
- **Neural Network (MLP)** won Rank 1 on CV performance (CV ROC-AUC: `0.9925`, CV Malignant Recall: `0.9412`, CV Accuracy: `0.9516`).
- It was selected as champion without touching `X_test`.
- Subsequent evaluation on the untouched holdout test set confirmed strong generalization (Test ROC-AUC: `0.9937`, Test Malignant Recall: `0.9524`, Test Accuracy: `0.9737`).

---

## Summary of Completed Refactoring
1. **Reproducibility**: `train_model.py` trains all models from scratch and reproduces all benchmark figures deterministically.
2. **Zero Test Leakage**: Hyperparameter tuning AND champion model selection are executed exclusively on `X_train` cross-validation.
3. **Correctness**: Scikit-learn label convention ($0 = \text{Malignant}$) is strictly enforced; Malignant Recall is prioritized.
4. **Architecture**: Streamlit + scikit-learn Pipeline replaces the previous mocked client-side JavaScript.
5. **Educational Value**: Comprehensive mathematical and technical guide in `docs/PROJECT_GUIDE.md`.

