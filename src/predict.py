from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Load the pipeline saved during training.
model = joblib.load(
    ROOT / "models" / "best_phishing_model.joblib"
)

# Read the exact feature names and order used during training.
features = pd.read_csv(
    ROOT / "models" / "feature_columns.csv"
)["feature"].tolist()

df = pd.read_csv(ROOT / "data" / "dataset.csv")
df.columns = df.columns.str.strip()

# Select 10 records to demonstrate prediction.
samples = df.sample(n=10, random_state=7)
X = samples.loc[:, features]

predictions = model.predict(X)

names = {0: "Legitimate", 1: "Phishing"}
actual = (samples["Result"] == -1).astype(int)

results = pd.DataFrame({
    "Dataset row": samples.index,
    "Actual": actual.map(names).to_numpy(),
    "Predicted": [names[int(value)] for value in predictions],
})

results["Correct"] = results["Actual"] == results["Predicted"]

print("\nSaved-model prediction demonstration\n")
print(results.to_string(index=False))
print(
    "\nNote: these samples come from the original dataset and "
    "may include training records. This is a functionality "
    "demonstration, not a new accuracy evaluation."
)

output = ROOT / "reports" / "sample_predictions.csv"
output.parent.mkdir(exist_ok=True)
results.to_csv(output, index=False)

print("\nSaved results to reports/sample_predictions.csv")