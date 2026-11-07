
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

st.set_page_config(
    page_title="ML Classification Model Evaluator",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Model Evaluator")

st.write(
    "Upload a labelled test CSV to evaluate five pre-trained "
    "classification models. No model retraining is performed."
)

# --------------------------------------------------
# 2. Load model metadata
# --------------------------------------------------

with open(MODEL_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

model_names = list(metadata["models"].keys())
feature_names = metadata["feature_names"]
target_col = metadata["target_column"]

# --------------------------------------------------
# 3. Upload test dataset
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"]
)

st.caption(
    f"Required input: {len(feature_names)} features and target "
    f"column '{target_col}'. Target mapping: 0 = Benign, 1 = Malignant."
)

# --------------------------------------------------
# 4. Evaluate uploaded data
# --------------------------------------------------

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

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

        elif data.empty:
            st.error("The uploaded CSV contains no rows.")

        else:
            X_test = data[feature_names].apply(
                pd.to_numeric, errors="coerce"
            )

            y_test = pd.to_numeric(
                data[target_col], errors="coerce"
            )

            if X_test.isna().any().any() or y_test.isna().any():
                st.error(
                    "Features and target must contain valid numeric "
                    "values without missing entries."
                )

            elif not np.isfinite(X_test.to_numpy()).all() or not np.isfinite(y_test.to_numpy()).all():
                st.error("The uploaded data contains infinite values.")

            elif not set(y_test.unique()).issubset({0, 1}):
                st.error("Target values must be 0 (Benign) or 1 (Malignant).")

            else:
                y_test = y_test.astype(int)

                # AUC requires both target classes.
                auc_available = len(y_test.unique()) == 2

                # --------------------------------------------------
                # 5. Evaluate all five models
                # --------------------------------------------------

                all_results = {}
                all_predictions = {}

                for name, filename in metadata["models"].items():

                    model_path = MODEL_DIR / filename

                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Model file not found: {filename}"
                        )

                    model = joblib.load(model_path)

                    predictions = model.predict(X_test)

                    probabilities = model.predict_proba(X_test)[:, 1]

                    metrics = {
                        "Model": name,
                        "Accuracy": accuracy_score(
                            y_test, predictions
                        ),
                        "AUC": (
                            roc_auc_score(y_test, probabilities)
                            if auc_available else np.nan
                        ),
                        "Precision": precision_score(
                            y_test, predictions, zero_division=0
                        ),
                        "Recall": recall_score(
                            y_test, predictions, zero_division=0
                        ),
                        "F1": f1_score(
                            y_test, predictions, zero_division=0
                        ),
                        "MCC": matthews_corrcoef(
                            y_test, predictions
                        )
                    }

                    all_results[name] = metrics
                    all_predictions[name] = predictions

                # --------------------------------------------------
                # 6. Display comparison of all models
                # --------------------------------------------------

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
                    results_df["Accuracy"].idxmax(), "Model"
                ]

                st.success(
                    f"Highest accuracy on this uploaded dataset: {best_model}"
                )

                # --------------------------------------------------
                # 7. Select a model for detailed evaluation
                # --------------------------------------------------

                st.subheader("Detailed Model Evaluation")

                selected_model = st.selectbox(
                    "Select classification model",
                    model_names
                )

                selected_predictions = all_predictions[selected_model]

                selected_metrics = all_results[selected_model]

                metric_columns = st.columns(6)

                for col, metric in zip(
                    metric_columns,
                    [
                        "Accuracy", "AUC", "Precision",
                        "Recall", "F1", "MCC"
                    ]
                ):
                    value = selected_metrics[metric]

                    col.metric(
                        metric,
                        f"{value:.4f}" if pd.notna(value) else "N/A"
                    )

                # --------------------------------------------------
                # 8. Confusion matrix and classification report
                # --------------------------------------------------

                left, right = st.columns(2)

                with left:

                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y_test,
                        selected_predictions,
                        labels=[0, 1]
                    )

                    fig, ax = plt.subplots()

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
                # 9. Preview uploaded data
                # --------------------------------------------------

                with st.expander("Preview uploaded test data"):

                    st.dataframe(
                        data.head(10),
                        use_container_width=True
                    )

    except Exception as exc:

        st.error(
            f"Could not evaluate the uploaded file: {exc}"
        )

else:

    st.info(
        "Upload the provided test_data.csv file to view "
        "the evaluation results."
    )
