# Phishing Website Detector

A machine learning project that classifies website records as phishing or
legitimate using 30 pre-engineered website features.

The project compares Logistic Regression and Random Forest, evaluates both
models, and saves the selected model with its preprocessing pipeline.

## Dataset

Source:
https://www.kaggle.com/datasets/akashkr/phishing-website-dataset

Original dataset:
- 11,055 rows
- 30 predictive features
- One row-identifier column: `index`
- One target column: `Result`

Original target labels:
- `-1`: phishing
- `1`: legitimate

Labels used by this project:
- `1`: phishing
- `0`: legitimate

## Data Preparation

- Removed the `index` column from the model inputs.
- Checked for missing values: none were found.
- Removed 5,206 identical feature-and-label records.
- Retained 5,849 records after deduplication.
- Identified 64 feature patterns associated with both target labels.
- Retained these conflicting-label records without guessing their correct labels.
- Kept identical feature patterns together during splitting to prevent them
  from appearing across training, validation, and test partitions.

Class distribution after cleaning:
- Legitimate: 2,830
- Phishing: 3,019

## Evaluation Method

Used group-based splits with a fixed random seed of 42:

| Partition | Rows |
|---|---:|
| Training | 3,509 |
| Validation | 1,167 |
| Test | 1,173 |

Models were initially fitted on the training partition.

The winner was selected using validation phishing recall, with precision and
accuracy as tie-breakers. Both models were then refitted on training plus
validation data and evaluated on the held-out test partition.

The selected model was not changed based on test results.

Preprocessing was included inside each model's pipeline.

## Demonstrate Saved-Model Predictions

Run:

```powershell
.\.venv\Scripts\python.exe src/predict.py
```

The script loads the saved model, predicts labels for 10 sampled dataset
records, and prints actual labels alongside predictions.

Results are saved to `reports/sample_predictions.csv`.

These samples may include training records. This is a functionality
demonstration, not an independent accuracy evaluation.

### Logistic Regression

- Most-frequent-value imputation
- Standard scaling
- Balanced class weights
- Maximum 2,000 iterations

### Random Forest

- Most-frequent-value imputation
- 200 trees
- Balanced class weights
- Random seed: 42

## Validation Results

Precision, recall, and F1 refer to the phishing class.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 90.92% | 91.60% | 90.36% | 90.97% |
| Random Forest | 95.03% | 94.94% | 95.26% | 95.10% |

Random Forest was selected based on validation performance.

## Final Test Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 91.56% | 91.80% | 91.95% | 91.88% |
| Random Forest | 95.82% | 96.51% | 95.40% | 95.95% |

## Confusion Matrices

After training, open `reports/confusion_matrices.png` to inspect the
confusion matrices for both models.

## Project Structure

```text
phishing-website-detection-ml/
├── data/
│   └── dataset.csv
├── models/
│   ├── best_phishing_model.joblib
│   └── feature_columns.csv
├── notebooks/
│   └── phishing_detection.ipynb
├── reports/
│   ├── confusion_matrices.png
│   ├── results_summary.md
│   ├── test_metrics.csv
│   └── validation_metrics.csv
├── src/
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

The notebook is an optional exploration scaffold. The complete training
workflow is implemented in `src/train.py`.

## Run Locally on Windows

Run the following commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/train.py
```

Download the dataset separately and place it at `data/dataset.csv`
before running training.

## Generated Outputs

- `models/best_phishing_model.joblib`: selected Random Forest pipeline
- `models/feature_columns.csv`: required input feature names and order
- `reports/validation_metrics.csv`: validation comparison
- `reports/test_metrics.csv`: final test comparison
- `reports/confusion_matrices.png`: confusion matrix plots
- `reports/results_summary.md`: generated results summary

## Limitations

- This is a feature-based classifier, not a complete URL-scanning application.
- It requires the same 30 engineered features used during training.
- Raw URL feature extraction is not implemented.
- The dataset contains feature patterns with conflicting labels.
- Deduplication changes the evaluation population: results describe the
  cleaned records, not the original frequency-weighted dataset.
- Feature-grouped evaluation does not establish performance on new domains,
  future phishing campaigns, or live browsing traffic.
- The project is an educational prototype, not a production security tool.

## Future Improvements

- Cross-validation and hyperparameter tuning
- Analysis of false positives and false negatives
- Evaluation on a newer, independent dataset
- A consistent feature-extraction pipeline for new websites