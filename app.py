
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"

st.set_page_config(
    page_title="ML Classification Model Evaluator",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Model Evaluator")

st.write(
    "Evaluate five pre-trained classification models "
    "using the default test dataset or your own uploaded CSV. "
    "No model retraining is performed."
)

# --------------------------------------------------
# 2. Load model metadata
# --------------------------------------------------

@st.cache_data
def load_metadata():
    with open(MODEL_DIR / "metadata.json", "r") as f:
        return json.load(f)


metadata = load_metadata()

model_names = list(metadata["models"].keys())
feature_names = metadata["feature_names"]
target_col = metadata["target_column"]

# --------------------------------------------------
# 3. Load all five pre-trained models
# --------------------------------------------------

@st.cache_resource
def load_models():

    models = {}

    for name, filename in metadata["models"].items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {filename}"
            )

        models[name] = joblib.load(model_path)

    return models


# --------------------------------------------------
# 4. Dataset selection
# --------------------------------------------------

st.subheader("Test Dataset")

uploaded_file = st.file_uploader(
    "Upload another test dataset (CSV) - Optional",
    type=["csv"]
)

try:

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.info(
            f"Using uploaded dataset: {uploaded_file.name}"
        )

    else:

        if not TEST_DATA_PATH.exists():
            st.error(
                "Default test_data.csv not found. "
                "Please upload a test CSV file."
            )
            st.stop()

        data = pd.read_csv(TEST_DATA_PATH)

        st.success(
            "Default test_data.csv loaded automatically."
        )

except Exception as exc:

    st.error(f"Unable to load dataset: {exc}")
    st.stop()


st.caption(
    f"Required input: {len(feature_names)} features "
    f"and target column '{target_col}'. "
    "Target mapping: 0 = Benign, 1 = Malignant."
)

# --------------------------------------------------
# 5. Validate dataset
# --------------------------------------------------

required_columns = feature_names + [target_col]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if data.empty:

    st.error("The dataset contains no rows.")
    st.stop()


X_test = data[feature_names].apply(
    pd.to_numeric,
    errors="coerce"
)

y_test = pd.to_numeric(
    data[target_col],
    errors="coerce"
)

if X_test.isna().any().any() or y_test.isna().any():

    st.error(
        "Features and target must contain valid numeric "
        "values without missing entries."
    )
    st.stop()

if (
    not np.isfinite(X_test.to_numpy()).all()
    or not np.isfinite(y_test.to_numpy()).all()
):

    st.error("Dataset contains infinite values.")
    st.stop()

if not set(y_test.unique()).issubset({0, 1}):

    st.error(
        "Target values must be 0 (Benign) or 1 (Malignant)."
    )
    st.stop()

y_test = y_test.astype(int)

# --------------------------------------------------
# 6. Dataset information
# --------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(data))
col2.metric("Total Features", len(feature_names))
col3.metric("Target Classes", y_test.nunique())

with st.expander("Preview Test Dataset"):

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

# --------------------------------------------------
# 7. Load models and evaluate all five
# --------------------------------------------------

try:

    models = load_models()

    all_results = {}
    all_predictions = {}

    auc_available = len(y_test.unique()) == 2

    for name, model in models.items():

        predictions = model.predict(X_test)

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_test)[:, 1]

        elif hasattr(model, "decision_function"):

            probabilities = model.decision_function(X_test)

        else:

            probabilities = None

        if probabilities is not None and auc_available:

            auc = roc_auc_score(
                y_test,
                probabilities
            )

        else:

            auc = np.nan

        metrics = {

            "Model": name,

            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "AUC": auc,

            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "MCC": matthews_corrcoef(
                y_test,
                predictions
            )
        }

        all_results[name] = metrics
        all_predictions[name] = predictions

    # --------------------------------------------------
    # 8. Comparison of all five models
    # --------------------------------------------------

    st.divider()

    st.subheader("Comparison of All Five Models")

    results_df = pd.DataFrame(
        list(all_results.values())
    )

    st.dataframe(
        results_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    best_model = results_df.loc[
        results_df["Accuracy"].idxmax(),
        "Model"
    ]

    st.success(
        f"Best model based on Accuracy: {best_model}"
    )

    # --------------------------------------------------
    # 9. Model selection dropdown
    # --------------------------------------------------

    st.divider()

    st.subheader("Detailed Model Evaluation")

    selected_model = st.selectbox(
        "Select Classification Model",
        model_names
    )

    selected_predictions = all_predictions[selected_model]

    selected_metrics = all_results[selected_model]

    st.write(
        f"Evaluation results for **{selected_model}**"
    )

    # --------------------------------------------------
    # 10. Display evaluation metrics
    # --------------------------------------------------

    metric_columns = st.columns(6)

    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]

    for col, metric in zip(
        metric_columns,
        metric_names
    ):

        value = selected_metrics[metric]

        col.metric(
            metric,
            f"{value:.4f}" if pd.notna(value) else "N/A"
        )

    # --------------------------------------------------
    # 11. Confusion Matrix
    # --------------------------------------------------

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            selected_predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Benign (0)",
                "Malignant (1)"
            ]
        ).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )

        plt.xticks(rotation=15)

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------
    # 12. Classification Report
    # --------------------------------------------------

    with right:

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            selected_predictions,
            labels=[0, 1],
            target_names=[
                "Benign",
                "Malignant"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

    # --------------------------------------------------
    # 13. Download evaluation results
    # --------------------------------------------------

    st.divider()

    st.subheader("Download Results")

    csv_results = results_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Model Comparison Results",
        data=csv_results,
        file_name="model_evaluation_results.csv",
        mime="text/csv"
    )

    st.success("Model evaluation completed successfully!")

except Exception as exc:

    st.error(
        f"Error during model evaluation: {exc}"
    )
