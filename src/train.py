from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# Load and validate the dataset.
df = pd.read_csv(ROOT / "data" / "dataset.csv")
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

df = df.drop(columns=["index"], errors="ignore")

print("Missing values:", int(df.isna().sum().sum()))

if df["Result"].isna().any():
    raise ValueError("Result contains missing labels.")

if set(df["Result"].unique()) != {-1, 1}:
    raise ValueError(
        f"Unexpected target labels: {df['Result'].unique()}"
    )

# Remove identical records after removing the row identifier.
duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Duplicate rows removed:", duplicates)

X = df.drop(columns="Result").apply(pd.to_numeric, errors="raise")

# Count feature patterns associated with both labels.
feature_columns = X.columns.tolist()

conflicting_patterns = (
    df.groupby(feature_columns, dropna=False)["Result"]
    .nunique()
    .gt(1)
    .sum()
)

print("Feature patterns with conflicting labels:",
      int(conflicting_patterns))

# Keep the labels as supplied; do not guess which label is correct.
# Dataset convention: -1 = phishing, 1 = legitimate.
y = (df["Result"] == -1).astype(int)

print("Cleaned shape:", df.shape)
print("Class counts (0=legitimate, 1=phishing):")
print(y.value_counts().sort_index())

# Keep identical feature patterns in the same partition.
from sklearn.model_selection import GroupShuffleSplit

groups = df.groupby(
    feature_columns,
    dropna=False,
    sort=False,
).ngroup()

outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

dev_indices, test_indices = next(
    outer_split.split(X, y, groups=groups)
)

X_dev = X.iloc[dev_indices]
y_dev = y.iloc[dev_indices]
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)

train_indices, val_indices = next(
    inner_split.split(
        X_dev,
        y_dev,
        groups=groups.iloc[dev_indices],
    )
)

X_train = X_dev.iloc[train_indices]
y_train = y_dev.iloc[train_indices]
X_val = X_dev.iloc[val_indices]
y_val = y_dev.iloc[val_indices]

for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    print(f"{name}: {len(labels)} rows")
    if labels.nunique() != 2:
        raise ValueError(
            f"{name} split does not contain both classes."
        )

candidates = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
}


def measure(name, actual, predicted):
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1": f1_score(actual, predicted, zero_division=0),
    }


# Select a winner using validation data only.
validation_rows = []

for name, model in candidates.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    validation_rows.append(
        measure(name, y_val, model.predict(X_val))
    )

validation = pd.DataFrame(validation_rows)
validation.to_csv(REPORTS / "validation_metrics.csv", index=False)

best_name = validation.sort_values(
    ["Recall", "Precision", "Accuracy"],
    ascending=False,
).iloc[0]["Model"]

print("\nValidation results:")
print(validation.to_string(index=False))
print("\nSelected model:", best_name)

# Refit both candidates on train + validation, then report test scores.
# The winner is already fixed and is not changed using test scores.
test_rows = []
fitted = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, candidate) in zip(axes, candidates.items()):
    model = clone(candidate)
    model.fit(X_dev, y_dev)
    prediction = model.predict(X_test)

    fitted[name] = model
    test_rows.append(measure(name, y_test, prediction))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        prediction,
        labels=[0, 1],
        display_labels=["Legitimate", "Phishing"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(name)

fig.tight_layout()
fig.savefig(REPORTS / "confusion_matrices.png", dpi=200)
plt.close(fig)

test_results = pd.DataFrame(test_rows)
test_results.to_csv(REPORTS / "test_metrics.csv", index=False)

# Save the evaluated winner, including its preprocessing.
joblib.dump(
    fitted[best_name],
    MODELS / "best_phishing_model.joblib",
)

# Save the exact input schema separately.
pd.Series(X.columns, name="feature").to_csv(
    MODELS / "feature_columns.csv", index=False
)

# Generate a results summary ready to copy into the README.
table = test_results.to_csv(index=False)
summary = (
    "# Training Results\n\n"
    f"- Rows after cleaning: {len(df)}\n"
    f"- Features: {X.shape[1]}\n"
    f"- Duplicate rows removed: {duplicates}\n"
    f"- Training rows: {len(X_train)}\n"
    f"- Validation rows: {len(X_val)}\n"
    f"- Test rows: {len(X_test)}\n"
    f"- Selected model: {best_name}\n\n"
    "Selection used validation phishing recall, with precision "
    "and accuracy as tie-breakers. Both models were then refitted "
    "on training + validation data before final test evaluation.\n\n"
    "Precision, recall and F1 refer to phishing (positive class).\n\n"
    "## Test metrics\n\n"
    f"```csv\n{table}```\n\n"
    "Limitations: these are random-split benchmark results, not "
    "proof of performance on newly encountered websites. "
    "Predictions require the same engineered features; "
    "this project does not extract features from raw URLs.\n"
)

(REPORTS / "results_summary.md").write_text(
    summary, encoding="utf-8"
)

print("\nFinal test results:")
print(test_results.to_string(index=False))
print("\nDONE: model and reports saved successfully.")