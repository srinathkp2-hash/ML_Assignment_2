
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Load the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
# Convert the original target to: 1 = malignant, 0 = benign.
y = (dataset.target == 0).astype(int)
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ])
}

rows = []
for display_name, estimator in models.items():
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]

    scores = {
        "Model": display_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }
    rows.append(scores)

    filename = display_name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(estimator, MODEL_DIR / filename)

metrics = pd.DataFrame(rows)
metrics.to_csv(BASE_DIR / "model_metrics.csv", index=False)

# Test file includes all 30 features and the target column, as required.
test_data = X_test.copy()
test_data["target"] = y_test
test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

metadata = {
    "dataset_name": "Wisconsin Diagnostic Breast Cancer",
    "target_column": "target",
    "target_mapping": {"0": "Benign", "1": "Malignant"},
    "feature_names": feature_names,
    "models": {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "support_vector_machine.joblib"
    }
}
with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Training complete.")
print(metrics.round(4).to_string(index=False))
print(f"\nTest data saved: {BASE_DIR / 'test_data.csv'}")
