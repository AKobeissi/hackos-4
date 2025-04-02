import os
from exp3 import train_and_predict

# Set paths to your data
train_path = os.path.join("dataset", "de_train_split.parquet")
test_path = os.path.join("dataset", "de_test_split.parquet")

# Create output directory
os.makedirs("results", exist_ok=True)

# Train models and generate predictions
gene_columns, mrrmse_score = train_and_predict(
    train_path=train_path,
    test_path=test_path,
    output_path='results/xgboost_submission.csv',
    eval_mode=True  # Set to True to calculate validation MRRMSE
)

print(f"Validation MRRMSE Score: {mrrmse_score:.6f}")
print("Done! Predictions saved to results/xgboost_submission.csv")
