import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

def mrrmse(y_true, y_pred):
    """Calculate Mean Rowwise Root Mean Squared Error"""
    rmse_per_sample = np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
    return np.mean(rmse_per_sample)

def process_data(train_df, test_df=None):
    """Process data for the competition with minimal feature engineering"""
    print("Processing data...")
    
    # Identify metadata columns
    metadata_cols = ['cell_type', 'sm_name']
    if 'id' in train_df.columns:
        metadata_cols.append('id')
    
    # Extract cell types
    cell_types = train_df['cell_type'].unique()
    print(f"Found {len(cell_types)} unique cell types")
    
    # Encode cell types
    cell_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cell_encoder.fit(np.array(cell_types).reshape(-1, 1))
    
    # Get all unique small molecules (including combinations)
    sm_names = []
    for sm in train_df['sm_name'].unique():
        if '|' in sm:
            combinations = sm.split('|')
            for c in combinations:
                if c not in sm_names:
                    sm_names.append(c)
        else:
            if sm not in sm_names:
                sm_names.append(sm)
    
    print(f"Found {len(sm_names)} unique small molecules")
    
    # Function to encode small molecules
    def encode_small_molecules(sm_list):
        encoded_features = []
        for sm in sm_list:
            # Initialize zeros array
            sm_vector = np.zeros(len(sm_names))
            
            # Handle combination treatments
            if '|' in sm:
                combinations = sm.split('|')
                for c in combinations:
                    if c in sm_names:
                        idx = sm_names.index(c)
                        sm_vector[idx] = 1
            else:
                if sm in sm_names:
                    idx = sm_names.index(sm)
                    sm_vector[idx] = 1
            
            encoded_features.append(sm_vector)
        return np.array(encoded_features)
    
    # Identify gene expression columns (numeric columns)
    gene_columns = []
    for col in train_df.columns:
        if col not in metadata_cols:
            try:
                train_df[col].astype(float)
                gene_columns.append(col)
            except (ValueError, TypeError):
                print(f"Skipping non-numeric column: {col}")
    
    print(f"Found {len(gene_columns)} gene expression features")
    
    # Process training data features
    cell_train = np.array(train_df['cell_type']).reshape(-1, 1)
    cell_encoded_train = cell_encoder.transform(cell_train)
    sm_encoded_train = encode_small_molecules(train_df['sm_name'])
    
    # Extract target values
    y_train = train_df[gene_columns].astype(float).values
    
    # Calculate basic statistics for each sample
    train_stats = []
    for i in range(len(train_df)):
        sample = y_train[i]
        mean = np.mean(sample)
        std = np.std(sample)
        median = np.median(sample)
        train_stats.append([mean, std, median])
    
    train_stats = np.array(train_stats)
    
    # Scale the statistics
    stats_scaler = StandardScaler()
    train_stats_scaled = stats_scaler.fit_transform(train_stats)
    
    # Count the number of molecules in combinations
    sm_counts = []
    for sm in train_df['sm_name']:
        if '|' in sm:
            count = len(sm.split('|'))
        else:
            count = 1
        sm_counts.append([count])
    
    sm_counts = np.array(sm_counts)
    
    # Combine features
    X_train = np.hstack([cell_encoded_train, sm_encoded_train, train_stats_scaled, sm_counts])
    
    # Process test data if provided
    X_test = None
    if test_df is not None:
        # Process test features
        cell_test = np.array(test_df['cell_type']).reshape(-1, 1)
        cell_encoded_test = cell_encoder.transform(cell_test)
        sm_encoded_test = encode_small_molecules(test_df['sm_name'])
        
        # For test data, we need to estimate statistics based on cell type and small molecule
        cell_sm_stats = {}
        for i, row in train_df.iterrows():
            cell = row['cell_type']
            sm = row['sm_name']
            key = (cell, sm)
            
            if key not in cell_sm_stats:
                cell_sm_stats[key] = []
            
            cell_sm_stats[key].append(train_stats[i])
        
        # Average the statistics for each combination
        for key in cell_sm_stats:
            cell_sm_stats[key] = np.mean(cell_sm_stats[key], axis=0)
        
        # Apply to test data
        test_stats = []
        for i, row in test_df.iterrows():
            cell = row['cell_type']
            sm = row['sm_name']
            key = (cell, sm)
            
            if key in cell_sm_stats:
                test_stats.append(cell_sm_stats[key])
            else:
                # Use cell-type average if combination not seen
                cell_keys = [(c, s) for c, s in cell_sm_stats.keys() if c == cell]
                if cell_keys:
                    cell_stats = np.mean([cell_sm_stats[k] for k in cell_keys], axis=0)
                    test_stats.append(cell_stats)
                else:
                    # Fallback to overall mean
                    test_stats.append(np.mean(train_stats, axis=0))
        
        test_stats = np.array(test_stats)
        test_stats_scaled = stats_scaler.transform(test_stats)
        
        # Count molecules in test combinations
        test_sm_counts = []
        for sm in test_df['sm_name']:
            if '|' in sm:
                count = len(sm.split('|'))
            else:
                count = 1
            test_sm_counts.append([count])
        
        test_sm_counts = np.array(test_sm_counts)
        
        # Combine test features
        X_test = np.hstack([cell_encoded_test, sm_encoded_test, test_stats_scaled, test_sm_counts])
    
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'gene_columns': gene_columns
    }
    
    if test_df is not None and 'id' in test_df.columns:
        result['test_ids'] = test_df['id'].values
    
    return result

def train_and_predict(train_path, test_path, output_path='submission.csv', eval_mode=False):
    """Train XGBoost models and generate predictions"""
    # 1. Load data
    print("Loading data...")
    if train_path.endswith('.parquet'):
        train_df = pd.read_parquet(train_path)
    else:
        train_df = pd.read_csv(train_path)
    
    if test_path.endswith('.parquet'):
        test_df = pd.read_parquet(test_path)
    else:
        test_df = pd.read_csv(test_path)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # 2. Process data
    data = process_data(train_df, test_df)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, gene_columns = data['X_test'], data['gene_columns']
    
    # 3. Train models - one per gene
    print(f"Training {y_train.shape[1]} XGBoost models, one per gene...")
    
    if eval_mode:
        # Split data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        train_data = X_train_split
        train_labels = y_train_split
    else:
        train_data = X_train
        train_labels = y_train
    
    predictions = np.zeros((X_test.shape[0], y_train.shape[1]))
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',  # Faster algorithm
        'silent': 1
    }
    
    # Train in batches to avoid memory issues
    batch_size = 1000  # Adjust based on your available memory
    n_genes = y_train.shape[1]
    batch_predictions = []
    
    for batch_start in range(0, n_genes, batch_size):
        batch_end = min(batch_start + batch_size, n_genes)
        print(f"Training genes {batch_start+1} to {batch_end} of {n_genes}...")
        
        batch_models = []
        for i in range(batch_start, batch_end):
            # Train a model for each gene
            dtrain = xgb.DMatrix(train_data, label=train_labels[:, i])
            
            # Different number of rounds based on importance of the gene
            # More rounds for genes with higher variance
            gene_std = np.std(train_labels[:, i])
            n_rounds = 300 if gene_std > np.median(np.std(train_labels, axis=0)) else 200
            
            model = xgb.train(params, dtrain, num_boost_round=n_rounds)
            batch_models.append(model)
        
        # Generate predictions for this batch
        for i, model in enumerate(batch_models):
            gene_idx = batch_start + i
            dtest = xgb.DMatrix(X_test)
            predictions[:, gene_idx] = model.predict(dtest)
        
        # Free up memory
        del batch_models
        import gc
        gc.collect()
    
    # 4. Save predictions
    submission_df = pd.DataFrame(
        predictions,
        columns=gene_columns
    )
    
    # Add IDs if available
    if 'test_ids' in data:
        submission_df.insert(0, 'id', data['test_ids'])
    
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # 5. Evaluate if requested
    mrrmse_score = None
    if eval_mode:
        # Predict on validation set
        val_predictions = np.zeros((X_val.shape[0], y_val.shape[1]))
        
        for batch_start in range(0, n_genes, batch_size):
            batch_end = min(batch_start + batch_size, n_genes)
            print(f"Validating genes {batch_start+1} to {batch_end} of {n_genes}...")
            
            batch_models = []
            for i in range(batch_start, batch_end):
                # Train a model for each gene
                dtrain = xgb.DMatrix(train_data, label=train_labels[:, i])
                
                gene_std = np.std(train_labels[:, i])
                n_rounds = 300 if gene_std > np.median(np.std(train_labels, axis=0)) else 200
                
                model = xgb.train(params, dtrain, num_boost_round=n_rounds)
                batch_models.append(model)
            
            # Generate validation predictions for this batch
            for i, model in enumerate(batch_models):
                gene_idx = batch_start + i
                dval = xgb.DMatrix(X_val)
                val_predictions[:, gene_idx] = model.predict(dval)
            
            # Free memory
            del batch_models
            gc.collect()
        
        # Calculate MRRMSE
        mrrmse_score = mrrmse(y_val, val_predictions)
        print(f"Validation MRRMSE: {mrrmse_score:.6f}")
    
    return gene_columns, mrrmse_score if eval_mode else gene_columns

if __name__ == "__main__":
    train_path = os.path.join("dataset", "de_train_split.parquet")
    test_path = os.path.join("dataset", "de_test_split.parquet")
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Train models and generate predictions
    gene_columns, mrrmse_score = train_and_predict(
        train_path=train_path,
        test_path=test_path,
        output_path='results/xgboost_submission.csv',
        eval_mode=True
    )
    
    print(f"Final validation MRRMSE: {mrrmse_score:.6f}")
    print("Done! Predictions saved to results/xgboost_submission.csv")
