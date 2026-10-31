# Training Results

- Rows after cleaning: 5849
- Features: 30
- Duplicate rows removed: 5206
- Training rows: 3509
- Validation rows: 1167
- Test rows: 1173
- Selected model: Random Forest

Selection used validation phishing recall, with precision and accuracy as tie-breakers. Both models were then refitted on training + validation data before final test evaluation.

Precision, recall and F1 refer to phishing (positive class).

## Test metrics

```csv
Model,Accuracy,Precision,Recall,F1
Logistic Regression,0.9156010230179028,0.9180327868852459,0.9195402298850575,0.918785890073831
Random Forest,0.9582267689684569,0.9651162790697675,0.9540229885057471,0.9595375722543352
```

Limitations: these are random-split benchmark results, not proof of performance on newly encountered websites. Predictions require the same engineered features; this project does not extract features from raw URLs.
