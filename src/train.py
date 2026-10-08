"""Model training, hyperparameter tuning, and cross-validation module.

Trains and compares 7 distinct classification paradigms:
    1. Logistic Regression (Linear probabilistic)
    2. K-Nearest Neighbors (Instance-based non-parametric)
    3. Decision Tree (Single rule-based CART)
    4. Random Forest (Bagging ensemble)
    5. Support Vector Machine (RBF Kernel with Platt scaling)
    6. Gradient Boosting (Sequential gradient-boosted ensemble)
    7. Multi-Layer Perceptron (Feed-forward Neural Network)

Both pipelines are evaluated under identical Stratified 5-Fold CV:
    - 15-Feature Production Pipeline (scaled, user-facing)
    - 30-Feature Experimental Baseline Pipeline
"""

from typing import Dict, Any, Tuple, List
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_loader import MALIGNANT_LABEL, BENIGN_LABEL
from src.preprocessing import create_production_pipeline, create_experimental_30_pipeline
from src.evaluate import evaluate_predictions, compute_roc_pr_curves


# Candidate model definitions and hyperparameter grids
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
        "description": "Linear probabilistic classifier using logistic sigmoid mapping and L2 regularization.",
    },
    "K-Nearest Neighbors": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "classifier__n_neighbors": [3, 5, 7, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"],
        },
        "description": "Non-parametric instance-based classifier assigning class by majority vote of nearest neighbors.",
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "classifier__max_depth": [3, 5, 7, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__criterion": ["gini", "entropy"],
        },
        "description": "Rule-based hierarchical classifier recursively partitioning feature space.",
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__max_features": ["sqrt", "log2"],
        },
        "description": "Bagging ensemble of de-correlated decision trees with randomized feature subsets.",
    },
    "Support Vector Machine": {
        "estimator": SVC(kernel="rbf", probability=True, random_state=42),
        "param_grid": {
            "classifier__C": [0.1, 1.0, 10.0, 50.0],
            "classifier__gamma": ["scale", "auto", 0.01],
        },
        "description": "Maximum-margin classifier projecting features into Hilbert space via Radial Basis Function (RBF).",
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "classifier__n_estimators": [100, 150],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 4],
        },
        "description": "Sequential boosting ensemble iteratively fitting decision trees to negative gradient residuals.",
    },
    "Neural Network (MLP)": {
        "estimator": MLPClassifier(max_iter=1000, random_state=42),
        "param_grid": {
            "classifier__hidden_layer_sizes": [(100,), (100, 50), (50, 25)],
            "classifier__alpha": [0.0001, 0.001, 0.01],
            "classifier__activation": ["relu"],
        },
        "description": "Multi-layer feed-forward artificial neural network trained via Adam backpropagation.",
    },
}


from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict


def train_and_tune_model(
    model_name: str,
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any], float]:
    """Tunes a pipeline using Stratified K-Fold GridSearchCV strictly on training data.

    Calculates out-of-fold cross-validation metrics on X_train to drive leak-free model selection.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Out-of-fold cross-validation predictions strictly on X_train (Zero test set access)
    y_cv_pred = cross_val_predict(best_pipeline, X_train, y_train, cv=cv)
    y_cv_proba = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method="predict_proba")
    cv_metrics = evaluate_predictions(y_train, y_cv_pred, y_cv_proba[:, MALIGNANT_LABEL])

    return best_pipeline, best_params, cv_metrics, elapsed_time


def evaluate_model_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates a fitted pipeline on holdout test data with Malignant as positive class.

    CRITICAL LEAKAGE RULE:
        This function must ONLY be called AFTER the champion model has been selected
        using training set cross-validation.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    y_proba_malignant = y_proba[:, MALIGNANT_LABEL]

    metrics = evaluate_predictions(y_test, y_pred, y_proba_malignant)
    curves = compute_roc_pr_curves(y_test, y_proba_malignant)
    return metrics, curves


def run_full_training_suite(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_features: List[str],
) -> Dict[str, Any]:
    """Trains and compares all 7 models following a strict leak-free protocol:

        80% training data (X_train)
            ↓
        Stratified 5-Fold CV & Hyperparameter Tuning
            ↓
        Compute CV Metrics on X_train
            ↓
        SELECT CHAMPION MODEL (Based strictly on CV performance on X_train)
            ↓
        Refit champion on full X_train (guaranteed by refit=True)
            ↓
        Evaluate ONCE on untouched 20% holdout test set (X_test)
    """
    X_tr_15 = X_train[selected_features].copy()
    X_te_15 = X_test[selected_features].copy()

    production_models: Dict[str, Any] = {}
    production_cv_metrics: Dict[str, Any] = {}
    production_test_metrics: Dict[str, Any] = {}

    baseline_30_models: Dict[str, Any] = {}
    baseline_30_cv_metrics: Dict[str, Any] = {}
    baseline_30_test_metrics: Dict[str, Any] = {}
    curves_data: Dict[str, Any] = {}

    print("\n" + "=" * 75)
    print("PHASE 1: HYPERPARAMETER TUNING & CROSS-VALIDATION ON TRAINING DATA ONLY")
    print(f"Training Samples: {len(X_train)} (Holdout Test Set is locked and untouched)")
    print(f"Production Features: {len(selected_features)}, Baseline Features: {X_train.shape[1]}")
    print("=" * 75)

    # 1. Train and tune all models strictly on X_train
    for name, config in MODEL_REGISTRY.items():
        print(f"\n---> Tuning on X_train CV: {name}")

        # A. 15-Feature Production Pipeline
        pipe_15 = create_production_pipeline(config["estimator"])
        best_pipe_15, best_params_15, cv_m_15, fit_time_15 = train_and_tune_model(
            name, pipe_15, config["param_grid"], X_tr_15, y_train
        )
        cv_m_15["best_params"] = best_params_15
        cv_m_15["training_seconds"] = round(fit_time_15, 3)
        cv_m_15["description"] = config["description"]

        production_models[name] = best_pipe_15
        production_cv_metrics[name] = cv_m_15

        print(
            f"     [15-Feat CV Score] CV ROC-AUC: {cv_m_15.get('roc_auc', 0):.4f} | "
            f"CV Malignant Recall: {cv_m_15['malignant_recall']:.4f} | "
            f"CV Accuracy: {cv_m_15['accuracy']:.4f}"
        )

        # B. 30-Feature Baseline Pipeline
        pipe_30 = create_experimental_30_pipeline(config["estimator"])
        best_pipe_30, best_params_30, cv_m_30, fit_time_30 = train_and_tune_model(
            name, pipe_30, config["param_grid"], X_train, y_train
        )
        cv_m_30["best_params"] = best_params_30
        cv_m_30["training_seconds"] = round(fit_time_30, 3)

        baseline_30_models[name] = best_pipe_30
        baseline_30_cv_metrics[name] = cv_m_30

    # 2. SELECT CHAMPION MODEL STRICTLY FROM TRAINING CROSS-VALIDATION
    print("\n" + "=" * 75)
    print("PHASE 2: CHAMPION MODEL SELECTION (BASED EXCLUSIVELY ON X_train CV SCORES)")
    print("Holdout test set has NOT been accessed or evaluated yet.")
    print("=" * 75)

    champion_name = select_champion_model(production_cv_metrics)
    champ_cv = production_cv_metrics[champion_name]
    print(
        f"\n>>> CHAMPION MODEL SELECTED: {champion_name}\n"
        f"    Based on Training Set 5-Fold Cross-Validation:\n"
        f"    - CV ROC-AUC: {champ_cv.get('roc_auc', 0):.4f}\n"
        f"    - CV Malignant Recall: {champ_cv['malignant_recall']:.4f}\n"
        f"    - CV Malignant F1: {champ_cv['malignant_f1']:.4f}\n"
        f"    - CV Accuracy: {champ_cv['accuracy']:.4f}"
    )

    # 3. NOW AND ONLY NOW: EVALUATE ON THE UNTOUCHED HOLDOUT TEST SET
    print("\n" + "=" * 75)
    print("PHASE 3: FINAL EVALUATION ON UNTOUCHED HOLDOUT TEST SET (N=114)")
    print("Evaluating models ONCE on the test set for unbiased generalization metrics.")
    print("=" * 75)

    for name in MODEL_REGISTRY:
        # Evaluate 15-Feature Production Model on Test Set
        pipe_15 = production_models[name]
        test_m_15, curves_15 = evaluate_model_on_test(pipe_15, X_te_15, y_test)
        test_m_15["best_params"] = production_cv_metrics[name]["best_params"]
        test_m_15["training_seconds"] = production_cv_metrics[name]["training_seconds"]
        test_m_15["description"] = production_cv_metrics[name]["description"]

        production_test_metrics[name] = test_m_15
        curves_data[name] = curves_15

        # Evaluate 30-Feature Baseline Model on Test Set
        pipe_30 = baseline_30_models[name]
        test_m_30, _ = evaluate_model_on_test(pipe_30, X_test, y_test)
        test_m_30["best_params"] = baseline_30_cv_metrics[name]["best_params"]
        test_m_30["training_seconds"] = baseline_30_cv_metrics[name]["training_seconds"]

        baseline_30_test_metrics[name] = test_m_30

        is_champ_tag = " [CHAMPION]" if name == champion_name else ""
        print(
            f"     {name + is_champ_tag:<30} | Test AUC: {test_m_15.get('roc_auc', 0):.4f} | "
            f"Test Recall: {test_m_15['malignant_recall']:.4f} | "
            f"Test Acc: {test_m_15['accuracy']:.4f}"
        )

    return {
        "production_models": production_models,
        "production_cv_metrics": production_cv_metrics,
        "production_test_metrics": production_test_metrics,
        "baseline_30_cv_metrics": baseline_30_cv_metrics,
        "baseline_30_test_metrics": baseline_30_test_metrics,
        "curves_data": curves_data,
        "champion_name": champion_name,
        "selected_features": selected_features,
    }


def select_champion_model(cv_metrics_dict: Dict[str, Dict[str, Any]]) -> str:
    """Selects the champion model based EXCLUSIVELY on training set cross-validation.

    SELECTION PROTOCOL:
        1. Evaluates out-of-fold CV scores on X_train.
        2. Prioritizes safety-oriented Malignant Recall (Sensitivity) alongside discrimination (ROC-AUC).
        3. Zero access to the holdout test set.
    """
    candidates = []
    for name, m in cv_metrics_dict.items():
        recall = m["malignant_recall"]
        auc = m.get("roc_auc", 0.0)
        f1 = m["malignant_f1"]
        # Ranking score based exclusively on cross-validation on X_train
        score = 0.40 * auc + 0.35 * recall + 0.25 * f1
        candidates.append((score, auc, recall, name))

    candidates.sort(reverse=True)
    return candidates[0][3]

