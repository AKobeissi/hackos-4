# Quick runner script for the advanced ensemble solution

import os
from essential_solution import train_and_predict

# Set paths to your data
train_path = os.path.join("dataset", "de_train_split.parquet")
test_path = os.path.join("dataset", "de_test_split.parquet")

# Create output directory
os.makedirs("results", exist_ok=True)

# Train the ensemble and generate predictions
models, gene_columns, mrrmse_score = train_and_predict(
    train_path=train_path,
    test_path=test_path,
    output_path='results/ensemble_submission.csv',
    eval_mode=True  # Enable MRRMSE evaluation
)

print(f"Validation MRRMSE Score: {mrrmse_score:.6f}")
print("Done! Predictions saved to results/ensemble_submission.csv")
