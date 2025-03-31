import os
import numpy as np
import pandas as pd
from train import train_and_predict  # Import the train_and_predict function from train.py

# Set paths to your data
train_path = os.path.join("dataset", "de_train_split.parquet")
test_path = os.path.join("dataset", "de_test_split.parquet")

# Create output directory
os.makedirs("results", exist_ok=True)

# Train model, generate predictions, and evaluate with MRRMSE
model, gene_columns, mrrmse_score = train_and_predict(
    train_path=train_path,
    test_path=test_path,
    output_path='results/submission.csv',
    eval_mode=True  # Enable MRRMSE calculation
)

print(f"Validation MRRMSE Score: {mrrmse_score:.6f}")
print("Done! Predictions saved to results/submission.csv")

# Run a cross-validation to get a more robust MRRMSE estimate
def run_cv_mrrmse(n_folds=5):
    from sklearn.model_selection import KFold

    train_df = pd.read_parquet(train_path)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mrrmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\nFold {fold+1}/{n_folds}")

        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_df.iloc[val_idx].reset_index(drop=True)

        train_fold_path = os.path.join("results", f"train_fold_{fold}.parquet")
        val_fold_path = os.path.join("results", f"val_fold_{fold}.parquet")

        train_fold.to_parquet(train_fold_path)
        val_fold.to_parquet(val_fold_path)

        _, _, fold_mrrmse = train_and_predict(
            train_path=train_fold_path,
            test_path=val_fold_path,
            output_path=os.path.join("results", f"fold_{fold}_preds.csv"),
            eval_mode=True
        )
        mrrmse_scores.append(fold_mrrmse)

        os.remove(train_fold_path)
        os.remove(val_fold_path)

    mean_mrrmse = np.mean(mrrmse_scores)
    std_mrrmse = np.std(mrrmse_scores)

    print("\nCross-validation results:")
    print(f"Mean MRRMSE: {mean_mrrmse:.6f} ± {std_mrrmse:.6f}")
    print(f"Individual fold scores: {mrrmse_scores}")

    return mean_mrrmse

cv_mrrmse = run_cv_mrrmse(n_folds=3)
