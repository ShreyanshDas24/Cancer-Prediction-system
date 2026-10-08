"""Comprehensive model evaluation module.

CRITICAL LABEL CONVENTION:
    In scikit-learn Wisconsin Breast Cancer dataset:
        - 0 = 'malignant'
        - 1 = 'benign'
    To evaluate safety-critical diagnostic performance, Malignant (label 0)
    is treated as the primary positive class of interest.

All metric calculations explicitly reference:
    - TP = Malignant correctly classified as Malignant
    - FN = Malignant incorrectly classified as Benign (critical error)
    - FP = Benign incorrectly classified as Malignant
    - TN = Benign correctly classified as Benign
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from src.data_loader import MALIGNANT_LABEL, BENIGN_LABEL


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_malignant: np.ndarray = None,
) -> Dict[str, Any]:
    """Calculates full suite of classification metrics with Malignant as positive class.

    Args:
        y_true: True labels (0 = malignant, 1 = benign).
        y_pred: Predicted labels (0 = malignant, 1 = benign).
        y_proba_malignant: Predicted probability of being Malignant (class 0).

    Returns:
        metrics (dict): Comprehensive evaluation metrics and confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Confusion matrix with explicit label ordering: [Malignant (0), Benign (1)]
    # cm[0, 0] = TP (True Malignant)
    # cm[0, 1] = FN (Malignant missed, predicted Benign)
    # cm[1, 0] = FP (Benign misdiagnosed as Malignant)
    # cm[1, 1] = TN (True Benign)
    cm = confusion_matrix(y_true, y_pred, labels=[MALIGNANT_LABEL, BENIGN_LABEL])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))

    # Recall on Malignant: TP / (TP + FN)
    malignant_recall = float(
        recall_score(y_true, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0)
    )

    # Specificity on Benign: TN / (TN + FP)
    benign_specificity = float(
        recall_score(y_true, y_pred, pos_label=BENIGN_LABEL, zero_division=0)
    )

    # Precision on Malignant: TP / (TP + FP)
    malignant_precision = float(
        precision_score(y_true, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0)
    )

    # F1-score for Malignant class
    malignant_f1 = float(
        f1_score(y_true, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0)
    )

    metrics = {
        "accuracy": acc,
        "malignant_recall": malignant_recall,
        "malignant_precision": malignant_precision,
        "malignant_f1": malignant_f1,
        "benign_specificity": benign_specificity,
        "confusion_matrix": {
            "tp_malignant": tp,
            "fn_malignant": fn,
            "fp_benign": fp,
            "tn_benign": tn,
            "raw_matrix": cm.tolist(),
        },
    }

    if y_proba_malignant is not None:
        y_true_binary = (y_true == MALIGNANT_LABEL).astype(int)
        roc_auc = float(roc_auc_score(y_true_binary, y_proba_malignant))
        metrics["roc_auc"] = roc_auc

    return metrics


def compute_roc_pr_curves(
    y_true: np.ndarray, y_proba_malignant: np.ndarray
) -> Dict[str, Any]:
    """Generates ROC and Precision-Recall curve coordinates for plotting."""
    y_true_binary = (np.asarray(y_true) == MALIGNANT_LABEL).astype(int)
    fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_proba_malignant)
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true_binary, y_proba_malignant
    )

    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def compute_educational_threshold_analysis(
    y_true: np.ndarray, y_proba_malignant: np.ndarray, steps: int = 19
) -> pd.DataFrame:
    """Educational demonstration of sensitivity vs specificity trade-offs.

    Varies the classification threshold tau:
        Predict Malignant if P(Malignant) >= tau.

    EDUCATIONAL DISCLAIMER:
        This analysis demonstrates mathematical trade-offs between false positives
        and false negatives. It is strictly educational and not a clinically validated decision rule.
    """
    thresholds = np.linspace(0.05, 0.95, steps)
    y_true_binary = (np.asarray(y_true) == MALIGNANT_LABEL).astype(int)

    rows = []
    for tau in thresholds:
        y_pred_binary = (y_proba_malignant >= tau).astype(int)

        tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
        fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
        fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
        tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        rows.append(
            {
                "threshold": round(float(tau), 3),
                "sensitivity_recall": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "precision": round(precision, 4),
                "false_negatives": fn,
                "false_positives": fp,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    y_t = np.array([0, 0, 1, 1, 0, 1])
    y_p = np.array([0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.85, 0.05])
    res = evaluate_predictions(y_t, y_p, y_prob)
    print("Sample Evaluation:")
    for k, v in res.items():
        print(f"  {k}: {v}")
