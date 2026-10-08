"""Advanced Cancer Prediction System - Interactive Machine Learning Application.

Built with Streamlit and scikit-learn.
Educational/Research Machine Learning Demonstration.

Target Convention:
    0 = Malignant (Positive Class of Interest)
    1 = Benign
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.predict import CancerPredictor
from src.data_loader import load_raw_data, MALIGNANT_LABEL, BENIGN_LABEL, LABEL_NAMES
from src.evaluate import compute_educational_threshold_analysis

# Page Configuration
st.set_page_config(
    page_title="Cancer Prediction System | Educational ML",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished typography, badges, and cards
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.5rem 0.85rem;
        border-radius: 6px;
        color: #64748B;
        font-size: 0.82rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-malignant {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .badge-benign {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #34D399;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    """Caches the predictor instance for fast inference."""
    return CancerPredictor()


@st.cache_data
def get_metrics_summary():
    """Loads precomputed model comparison metrics."""
    metrics_path = "models/metrics_summary.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def get_raw_data_cached():
    """Loads the cached dataset for EDA."""
    X, y, meta = load_raw_data()
    df = X.copy()
    df["target"] = y
    df["diagnosis"] = y.map({MALIGNANT_LABEL: "Malignant", BENIGN_LABEL: "Benign"})
    return df, meta


def main():
    predictor = get_predictor()
    metrics_data = get_metrics_summary()
    df_raw, raw_meta = get_raw_data_cached()

    # App Header
    st.markdown('<div class="main-header">🔬 Cancer Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Reproducible Machine Learning Pipeline & Comparative Diagnostics Study | Wisconsin Breast Cancer Dataset</div>',
        unsafe_allow_html=True,
    )

    # Educational Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            Educational project using the Wisconsin Diagnostic Breast Cancer dataset. Predictions are statistical model outputs and are not medical diagnoses.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Navigation Tabs
    tab_predict, tab_compare, tab_eda = st.tabs(
        [
            "🎯 Diagnostic Prediction",
            "📊 Model Comparison & Benchmarks",
            "🔍 Exploratory Data Analysis (EDA)",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: DIAGNOSTIC PREDICTION
    # -------------------------------------------------------------
    with tab_predict:
        st.subheader("Statistical Prediction & Risk Categorization")
        st.markdown(
            "Enter the **15 production features** (selected via leak-free ANOVA F-statistic) "
            "or load one of the clinical cohort presets."
        )

        # Model Selector & Presets Row
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            available_models = list(predictor.models_dict.keys())
            champion_model_name = metrics_data.get("champion_model", available_models[0]) if metrics_data else available_models[0]
            selected_model_name = st.selectbox(
                "Select ML Model for Prediction:",
                options=available_models,
                index=available_models.index(champion_model_name) if champion_model_name in available_models else 0,
                help="Choose from the 7 trained classification models.",
            )
            predictor.set_model(selected_model_name)

        with col_ctrl2:
            st.write("**Load Preset Sample:**")
            p_col1, p_col2, p_col3 = st.columns(3)

            preset_to_load = None
            if p_col1.button("🟢 Load Typical Benign Preset", use_container_width=True):
                preset_to_load = "benign"
            if p_col2.button("🔴 Load Typical Malignant Preset", use_container_width=True):
                preset_to_load = "malignant"
            if p_col3.button("🟡 Load Borderline Preset", use_container_width=True):
                preset_to_load = "borderline"

        # Organize 15 features into 3 logical categories
        features_by_cat = {"Size": [], "Shape": [], "Variability": []}
        for feat in predictor.selected_features:
            cat = predictor.metadata[feat]["category"]
            features_by_cat.get(cat, features_by_cat["Shape"]).append(feat)

        # Build Input Form
        input_values = {}
        with st.form("prediction_input_form"):
            st.markdown("##### Patient Biopsy Measurements (15 Features)")

            f_col1, f_col2, f_col3 = st.columns(3)

            # Category 1: Size
            with f_col1:
                st.markdown("**📏 Tumor Size Measurements**")
                for feat in features_by_cat["Size"]:
                    meta = predictor.metadata[feat]
                    default_val = meta["median"]
                    if preset_to_load == "benign":
                        default_val = meta["benign_median"]
                    elif preset_to_load == "malignant":
                        default_val = meta["malignant_median"]
                    elif preset_to_load == "borderline":
                        default_val = (meta["benign_median"] + meta["malignant_median"]) / 2.0

                    step_val = max(0.001, round((meta["max"] - meta["min"]) / 200, 3))
                    input_values[feat] = st.number_input(
                        f"{feat.title()} ({meta['unit']})",
                        min_value=float(meta["min"] * 0.5),
                        max_value=float(meta["max"] * 1.5),
                        value=float(default_val),
                        step=float(step_val),
                        format="%.3f" if step_val < 0.1 else "%.2f",
                        help=f"{meta['description']} | Benign Median: {meta['benign_median']:.2f}, Malignant Median: {meta['malignant_median']:.2f}",
                    )

            # Category 2: Shape
            with f_col2:
                st.markdown("**🔍 Tumor Shape & Margins**")
                for feat in features_by_cat["Shape"]:
                    meta = predictor.metadata[feat]
                    default_val = meta["median"]
                    if preset_to_load == "benign":
                        default_val = meta["benign_median"]
                    elif preset_to_load == "malignant":
                        default_val = meta["malignant_median"]
                    elif preset_to_load == "borderline":
                        default_val = (meta["benign_median"] + meta["malignant_median"]) / 2.0

                    step_val = max(0.0001, round((meta["max"] - meta["min"]) / 200, 4))
                    input_values[feat] = st.number_input(
                        f"{feat.title()} ({meta['unit']})",
                        min_value=float(meta["min"] * 0.5),
                        max_value=float(meta["max"] * 1.5),
                        value=float(default_val),
                        step=float(step_val),
                        format="%.4f" if step_val < 0.01 else "%.3f",
                        help=f"{meta['description']} | Benign Median: {meta['benign_median']:.3f}, Malignant Median: {meta['malignant_median']:.3f}",
                    )

            # Category 3: Variability
            with f_col3:
                st.markdown("**📊 Measurement Errors (SE)**")
                for feat in features_by_cat["Variability"]:
                    meta = predictor.metadata[feat]
                    default_val = meta["median"]
                    if preset_to_load == "benign":
                        default_val = meta["benign_median"]
                    elif preset_to_load == "malignant":
                        default_val = meta["malignant_median"]
                    elif preset_to_load == "borderline":
                        default_val = (meta["benign_median"] + meta["malignant_median"]) / 2.0

                    step_val = max(0.001, round((meta["max"] - meta["min"]) / 200, 3))
                    input_values[feat] = st.number_input(
                        f"{feat.title()} ({meta['unit']})",
                        min_value=float(meta["min"] * 0.5),
                        max_value=float(meta["max"] * 1.5),
                        value=float(default_val),
                        step=float(step_val),
                        format="%.3f" if step_val < 0.1 else "%.2f",
                        help=f"{meta['description']} | Benign Median: {meta['benign_median']:.2f}, Malignant Median: {meta['malignant_median']:.2f}",
                    )

            submit_prediction = st.form_submit_button("🚀 Run Prediction & Statistical Assessment", use_container_width=True)

        # Run Prediction and Display Results
        if submit_prediction or preset_to_load is not None:
            result = predictor.predict_sample(input_values)

            st.markdown("---")
            st.markdown("### 📋 Prediction Results & Assessment")

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)

            # Metric 1: Predicted Class
            with res_col1:
                st.markdown("**Predicted Class:**")
                if result["predicted_class"] == "Malignant":
                    st.markdown('<div class="badge-malignant">MALIGNANT (Class 0)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-benign">BENIGN (Class 1)</div>', unsafe_allow_html=True)

            # Metric 2: Malignant Probability
            with res_col2:
                st.metric(
                    label="Model Malignant Probability",
                    value=f"{result['proba_malignant'] * 100:.2f}%",
                    delta=f"{(result['proba_malignant'] - 0.5) * 100:.1f}% vs 50% threshold",
                    delta_color="inverse",
                )

            # Metric 3: Confidence Percentage
            with res_col3:
                st.metric(
                    label="Prediction Confidence",
                    value=f"{result['confidence_percentage']:.2f}%",
                    help="Probability assigned to the predicted class.",
                )

            # Metric 4: Risk Tier
            with res_col4:
                st.metric(
                    label="Model Risk Category",
                    value=result["model_risk_category"],
                    help="Statistical risk grouping based on probability of malignancy.",
                )

            # Probability Distribution Gauge Bar
            st.markdown("##### Probability Breakdown")
            prob_df = pd.DataFrame(
                {
                    "Class": ["Malignant (0)", "Benign (1)"],
                    "Probability": [result["proba_malignant"], result["proba_benign"]],
                    "Color": ["#EF4444", "#10B981"],
                }
            )
            fig_bar = px.bar(
                prob_df,
                x="Probability",
                y="Class",
                orientation="h",
                color="Class",
                color_discrete_map={"Malignant (0)": "#EF4444", "Benign (1)": "#10B981"},
                text=prob_df["Probability"].apply(lambda p: f"{p*100:.1f}%"),
                range_x=[0, 1],
            )
            fig_bar.update_layout(height=180, margin=dict(l=20, r=20, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Feature Deviation Analysis (Z-Scores vs. Benign Population)
            st.markdown("##### Feature Deviation Analysis (Relative to Benign Population Baseline)")
            st.caption(
                "Positive z-scores indicate values higher than the benign population mean. "
                "Higher positive deviations on size and shape features correlate strongly with malignancy."
            )

            dev_data = []
            for feat, d in result["feature_deviations"].items():
                dev_data.append(
                    {
                        "Feature": feat.title(),
                        "Z-Score vs Benign": d["z_score_vs_benign"],
                        "Patient Value": d["value"],
                        "Benign Median": d["benign_median"],
                        "Malignant Median": d["malignant_median"],
                        "Category": d["category"],
                    }
                )
            dev_df = pd.DataFrame(dev_data).sort_values("Z-Score vs Benign", ascending=True)

            fig_dev = px.bar(
                dev_df,
                x="Z-Score vs Benign",
                y="Feature",
                color="Z-Score vs Benign",
                color_continuous_scale="Reds",
                orientation="h",
                text=dev_df["Z-Score vs Benign"].apply(lambda z: f"{z:+.2f}σ"),
            )
            fig_dev.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig_dev, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: MODEL COMPARISON & BENCHMARKS
    # -------------------------------------------------------------
    with tab_compare:
        st.subheader("Model Performance Comparison & Benchmark Leaderboard")
        st.markdown(
            "Every model was evaluated using **Stratified 5-Fold Cross-Validation** on the training split ($N=455$) "
            "and evaluated on the unseen holdout test split ($N=114$). "
            "**Malignant (label 0) is treated as the primary positive class of interest.**"
        )

        if metrics_data:
            prod_metrics = metrics_data["production_15_metrics"]
            base_metrics = metrics_data["experimental_30_metrics"]

            # Leaderboard Table
            leaderboard_rows = []
            for name, m in prod_metrics.items():
                is_champ = name == metrics_data.get("champion_model")
                m_base = base_metrics.get(name, {})
                leaderboard_rows.append(
                    {
                        "Model": f"{name} 🏆 (CHAMPION)" if is_champ else name,
                        "15-Feat ROC-AUC": round(m.get("roc_auc", 0), 4),
                        "15-Feat Malignant Recall": round(m["malignant_recall"], 4),
                        "15-Feat Accuracy": round(m["accuracy"], 4),
                        "15-Feat Malignant Precision": round(m["malignant_precision"], 4),
                        "15-Feat Malignant F1": round(m["malignant_f1"], 4),
                        "15-Feat Benign Specificity": round(m["benign_specificity"], 4),
                        "30-Feat Baseline ROC-AUC": round(m_base.get("roc_auc", 0), 4),
                    }
                )

            leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("15-Feat ROC-AUC", ascending=False)
            st.dataframe(
                leaderboard_df.style.highlight_max(
                    subset=["15-Feat ROC-AUC", "15-Feat Malignant Recall", "15-Feat Accuracy", "15-Feat Malignant F1"],
                    color="#D1FAE5",
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Chart: 15-Feature Production vs 30-Feature Baseline Comparison
            st.markdown("##### 15-Feature Production Pipeline vs. 30-Feature Experimental Baseline (ROC-AUC)")
            comp_df = pd.DataFrame(
                [
                    {"Model": name, "Pipeline": "15-Feature Production", "ROC-AUC": prod_metrics[name].get("roc_auc", 0)}
                    for name in prod_metrics
                ]
                + [
                    {"Model": name, "Pipeline": "30-Feature Baseline", "ROC-AUC": base_metrics[name].get("roc_auc", 0)}
                    for name in base_metrics
                ]
            )
            fig_comp = px.bar(
                comp_df,
                x="Model",
                y="ROC-AUC",
                color="Pipeline",
                barmode="group",
                color_discrete_sequence=["#2563EB", "#93C5FD"],
                range_y=[0.90, 1.00],
            )
            fig_comp.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_comp, use_container_width=True)

            # ROC Curves and Confusion Matrix Inspection
            col_roc, col_cm = st.columns(2)

            with col_roc:
                st.markdown("##### Multi-Model ROC Curves (Test Set)")
                curves_data = metrics_data.get("curves_data", {})
                fig_roc = go.Figure()
                for name, curve in curves_data.items():
                    auc_val = prod_metrics[name].get("roc_auc", 0)
                    is_champ = name == metrics_data.get("champion_model")
                    fig_roc.add_trace(
                        go.Scatter(
                            x=curve["roc"]["fpr"],
                            y=curve["roc"]["tpr"],
                            name=f"{name} ({auc_val:.3f})",
                            mode="lines",
                            line=dict(width=3 if is_champ else 1.5, dash="solid" if is_champ else "dot"),
                        )
                    )
                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        name="Random Chance",
                        mode="lines",
                        line=dict(color="grey", dash="dash"),
                    )
                )
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate (1 - Specificity)",
                    yaxis_title="True Positive Rate (Malignant Recall)",
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(font=dict(size=9)),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_cm:
                st.markdown("##### Confusion Matrix Inspector")
                cm_model_choice = st.selectbox(
                    "Inspect Confusion Matrix for Model:",
                    options=list(prod_metrics.keys()),
                    index=list(prod_metrics.keys()).index(metrics_data.get("champion_model")),
                )
                cm_obj = prod_metrics[cm_model_choice]["confusion_matrix"]
                cm_matrix = np.array(cm_obj["raw_matrix"])

                fig_cm = px.imshow(
                    cm_matrix,
                    labels=dict(x="Predicted Diagnosis", y="True Diagnosis", color="Cases"),
                    x=["Pred Malignant (0)", "Pred Benign (1)"],
                    y=["True Malignant (0)", "True Benign (1)"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption(
                    f"**TP (Malignant correctly identified):** {cm_obj['tp_malignant']} | "
                    f"**FN (Malignant missed):** {cm_obj['fn_malignant']} | "
                    f"**FP (Benign false alarm):** {cm_obj['fp_benign']} | "
                    f"**TN (Benign correctly identified):** {cm_obj['tn_benign']}"
                )

            # Educational Sensitivity vs Specificity Threshold Analysis
            st.markdown("---")
            st.markdown("##### Educational Threshold Analysis (Sensitivity vs. Specificity Trade-Off)")
            st.markdown(
                "In binary classification with probabilistic models, changing the decision threshold $\\tau$ "
                "illustrates the direct trade-off between **Sensitivity (Malignant Recall)** and **Specificity (Benign Recall)**. "
                "*Lowering the threshold captures more malignant cases but increases false alarms.*"
            )

            # Generate threshold trade-off curve on test set for champion model
            champ_model = predictor.models_dict[metrics_data.get("champion_model")]
            X_test_15 = df_raw.iloc[int(len(df_raw)*0.8):][metrics_data["selected_features_15"]]
            y_test_raw = df_raw.iloc[int(len(df_raw)*0.8):]["target"]
            y_proba_champ = champ_model.predict_proba(X_test_15)[:, MALIGNANT_LABEL]

            thresh_df = compute_educational_threshold_analysis(y_test_raw.values, y_proba_champ, steps=19)

            fig_thresh = go.Figure()
            fig_thresh.add_trace(
                go.Scatter(
                    x=thresh_df["threshold"],
                    y=thresh_df["sensitivity_recall"],
                    name="Sensitivity (Malignant Recall)",
                    line=dict(color="#EF4444", width=2.5),
                )
            )
            fig_thresh.add_trace(
                go.Scatter(
                    x=thresh_df["threshold"],
                    y=thresh_df["specificity"],
                    name="Specificity (Benign Recall)",
                    line=dict(color="#10B981", width=2.5),
                )
            )
            fig_thresh.update_layout(
                xaxis_title="Classification Threshold (tau)",
                yaxis_title="Metric Score (0.0 - 1.0)",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_thresh, use_container_width=True)
            st.caption(
                "Educational note: Notice how at lower thresholds (e.g. 0.20), sensitivity rises while specificity drops. "
                "There is no 'zero false negative' threshold in realistic medical ML without causing unacceptable false positive rates."
            )

    # -------------------------------------------------------------
    # TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    with tab_eda:
        st.subheader("Exploratory Data Analysis & Multicollinearity Study")
        st.markdown(
            "The Wisconsin Diagnostic Breast Cancer (WDBC) dataset contains **569 instances** "
            "(357 Benign, 212 Malignant) with 30 continuous measurements."
        )

        eda_col1, eda_col2 = st.columns([1, 1])

        with eda_col1:
            st.markdown("##### Class Distribution (Target Imbalance)")
            class_counts = df_raw["diagnosis"].value_counts().reset_index()
            class_counts.columns = ["Diagnosis", "Count"]
            fig_pie = px.pie(
                class_counts,
                values="Count",
                names="Diagnosis",
                color="Diagnosis",
                color_discrete_map={"Malignant": "#EF4444", "Benign": "#10B981"},
                hole=0.45,
            )
            fig_pie.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("Class balance: 62.7% Benign vs. 37.3% Malignant.")

        with eda_col2:
            st.markdown("##### Feature Distribution Comparator")
            eda_feat = st.selectbox(
                "Select Feature to Compare:",
                options=predictor.selected_features,
                index=0,
            )
            fig_box = px.box(
                df_raw,
                x="diagnosis",
                y=eda_feat,
                color="diagnosis",
                color_discrete_map={"Malignant": "#EF4444", "Benign": "#10B981"},
                points="all",
            )
            fig_box.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=10, b=10),
                xaxis_title="Diagnosis",
                yaxis_title=f"{eda_feat.title()} ({predictor.metadata[eda_feat]['unit']})",
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Correlation Matrix Heatmap
        st.markdown("---")
        st.markdown("##### Feature Correlation Heatmap (Selected 15 Features)")
        st.markdown(
            "The dataset contains highly correlated geometric measurements (e.g., radius, perimeter, and area, where $r > 0.95$) "
            "because perimeter $\\approx 2\\pi r$ and area $\\approx \\pi r^2$. "
            "The production pipeline reduces the feature set from 30 to 15 using leak-free ANOVA F-statistic selection, "
            "substantially reducing input requirements while retaining strong classifier performance."
        )

        corr_matrix = df_raw[predictor.selected_features].corr()
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Pearson r"),
            x=[f.title() for f in predictor.selected_features],
            y=[f.title() for f in predictor.selected_features],
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig_corr.update_layout(height=600, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()
