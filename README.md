# ML Assignment 2 – Classification Models

## a. Problem statement

Build and compare five supervised classification algorithms on one public binary-classification dataset. Evaluate each model using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC), then provide an interactive Streamlit application that evaluates uploaded labelled test data using saved models.

## b. Dataset description

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), distributed with Scikit-learn and originating from the UCI Machine Learning Repository.

* 569 observations
* 30 numeric input features
* Binary target: 0 = Benign, 1 = Malignant
* No missing values in the supplied dataset
* A stratified 80:20 train/test split is used (`random\_state=42`).

The original Scikit-learn target is mapped so that malignant is the positive class (1). This is important when interpreting precision, recall, F1 and AUC.

## c. GitHub Repository Link

Adding public GitHub repository URL here:
`\[GITHUB REPOSITORY URL]`

## d. Models used

The following five models are trained on the same training split:

1. Logistic Regression (with StandardScaler)
2. Decision Tree Classifier
3. K-Nearest Neighbors (with StandardScaler)
4. Gaussian Naive Bayes
5. Support Vector Machine with RBF kernel (with StandardScaler)

### Evaluation metrics

The values below are from the included `train.py` run, using the fixed stratified test split. Results can vary if the split, parameters, or library versions are changed.

|Model|Accuracy|AUC|Precision|Recall|F1|MCC|
|-|-:|-:|-:|-:|-:|-:|
|Logistic Regression|0.9649|0.996|0.975|0.9286|0.9512|0.9245|
|Decision Tree|0.9211|0.9448|0.9459|0.8333|0.8861|0.8299|
|KNN|0.9561|0.9823|0.9744|0.9048|0.9383|0.9058|
|Naive Bayes|0.9386|0.9934|1|0.8333|0.9091|0.8715|
|Support Vector Machine|0.9737|0.9947|1|0.9286|0.963|0.9442|

### Observations on model performance

|ML Model Name|Observation about model performance|
|-|-|
|Logistic Regression|Strong overall performance with high accuracy and AUC. It provides a useful linear baseline after feature scaling.|
|Decision Tree|Lowest overall performance among these runs. Its rules are interpretable, but the selected tree settings may not capture the patterns as effectively.|
|KNN|Good performance after feature scaling. Its results depend on the choice of k and the distance calculation.|
|Naive Bayes|High AUC and perfect precision in this test split, but lower recall means some malignant cases were missed.|
|Support Vector Machine|Best accuracy, F1 and MCC in this test split, with perfect precision. It is the strongest overall model for this particular split.|

**Overall winner for this dataset:** Support Vector Machine for this particular train/test split. It has the highest accuracy (0.9737), F1 (0.9630) and MCC (0.9442). Logistic Regression is also highly competitive and has the highest AUC (0.9960). The winner is based on this split and should not be interpreted as proof that SVM will always perform best.

## Repository contents

* `app.py` – Streamlit front end
* `train.py` – data loading, split, training, evaluation and model persistence
* `requirements.txt` – Python dependencies
* `test\_data.csv` – held-out test data including target
* `model/` – five saved model pipelines/classifiers and metadata
* `model\_metrics.csv` – recorded test metrics

## Live Streamlit App Link

`\[STREAMLIT COMMUNITY CLOUD URL]`

