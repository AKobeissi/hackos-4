import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import os
import warnings
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Calculate MRRMSE
def mrrmse(y_true, y_pred):
    """
    Calculate Mean Rowwise Root Mean Squared Error
    
    Parameters:
    y_true (np.ndarray): True values
    y_pred (np.ndarray): Predicted values
    
    Returns:
    float: MRRMSE score
    """
    # Calculate RMSE for each row (sample)
    rmse_per_sample = np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
    
    # Calculate mean of all sample RMSEs
    mean_rmse = np.mean(rmse_per_sample)
    
    return mean_rmse

# Custom Dataset for PyTorch
class CellDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Advanced BiLSTM model with attention and residual connections
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(AttentionBiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Residual connection
        self.residual = nn.Linear(input_dim, hidden_dim * 2)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, hidden_dim)
        
        # Reshape for LSTM: (batch_size, seq_len=1, hidden_dim)
        embedded = embedded.unsqueeze(1)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, 1, hidden_dim*2)
        
        # Residual connection
        residual = self.residual(x).unsqueeze(1)  # (batch_size, 1, hidden_dim*2)
        lstm_out = lstm_out + residual
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # (batch_size, 1, 1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out)  # (batch_size, 1, hidden_dim*2)
        context = context.squeeze(1)  # (batch_size, hidden_dim*2)
        
        # Output layer
        output = self.fc(context)
        
        return output

# CNN model with residual connections
class ResidualCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.3):
        super(ResidualCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout)
        )
        
        # Reshape for CNN
        self.reshape_dim = hidden_dim * 2
        
        # CNN block 1
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection for block 1
        self.res1 = nn.Conv1d(1, hidden_dim, kernel_size=1)
        
        # CNN block 2
        self.cnn2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection for block 2
        self.res2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.reshape_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, hidden_dim*2)
        
        # Reshape for CNN: (batch_size, channels=1, features)
        cnn_input = embedded.view(batch_size, 1, self.reshape_dim)
        
        # First CNN block with residual connection
        res1 = self.res1(cnn_input)
        out1 = self.cnn1(cnn_input)
        out1 = out1 + res1
        
        # Second CNN block with residual connection
        res2 = self.res2(out1)
        out2 = self.cnn2(out1)
        out2 = out2 + res2
        
        # Global average pooling
        out = torch.mean(out2, dim=2)
        
        # Output layer
        output = self.fc(out)
        
        return output

# Data preprocessing function with advanced feature engineering
def preprocess_data(train_df, test_df=None, add_statistics=True):
    """
    Preprocess data with advanced feature engineering
    
    Parameters:
    train_df (pd.DataFrame): Training data
    test_df (pd.DataFrame): Test data (optional)
    add_statistics (bool): Whether to add statistical features
    
    Returns:
    dict: Dictionary with processed data
    """
    print("Preprocessing data with advanced feature engineering...")
    
    # Identify metadata columns
    metadata_cols = ['cell_type', 'sm_name']
    if 'id' in train_df.columns:
        metadata_cols.append('id')
    
    # Extract cell types
    cell_types = train_df['cell_type'].unique()
    print(f"Found {len(cell_types)} unique cell types")
    
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
    
    # One-hot encode cell types
    cell_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cell_encoder.fit(np.array(cell_types).reshape(-1, 1))
    
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
    
    # Process training data features
    cell_train = np.array(train_df['cell_type']).reshape(-1, 1)
    cell_encoded_train = cell_encoder.transform(cell_train)
    sm_encoded_train = encode_small_molecules(train_df['sm_name'])
    
    # Extract target values
    y_train = train_df[gene_columns].astype(float).values
    
    # Add statistical features if requested
    if add_statistics:
        print("Adding statistical features...")
        # Calculate statistics for each sample
        train_stats = []
        for i in range(len(train_df)):
            sample = y_train[i]
            mean = np.mean(sample)
            std = np.std(sample)
            q25 = np.percentile(sample, 25)
            median = np.median(sample)
            q75 = np.percentile(sample, 75)
            min_val = np.min(sample)
            max_val = np.max(sample)
            train_stats.append([mean, std, q25, median, q75, min_val, max_val])
        
        train_stats = np.array(train_stats)
        
        # Scale the statistics
        stats_scaler = StandardScaler()
        train_stats_scaled = stats_scaler.fit_transform(train_stats)
        
        # Combine features
        X_train = np.hstack([cell_encoded_train, sm_encoded_train, train_stats_scaled])
    else:
        # Combine features without statistics
        X_train = np.hstack([cell_encoded_train, sm_encoded_train])
    
    # Scale target values
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    
    # Process test data if provided
    X_test = None
    if test_df is not None:
        # Process test features
        cell_test = np.array(test_df['cell_type']).reshape(-1, 1)
        cell_encoded_test = cell_encoder.transform(cell_test)
        sm_encoded_test = encode_small_molecules(test_df['sm_name'])
        
        if add_statistics:
            # For test data, we need to estimate statistics
            # We'll use mean statistics for each cell type / small molecule combination
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
                    # If combination not seen in training, use overall mean
                    test_stats.append(np.mean(train_stats, axis=0))
            
            test_stats = np.array(test_stats)
            
            # Scale the statistics
            test_stats_scaled = stats_scaler.transform(test_stats)
            
            # Combine features
            X_test = np.hstack([cell_encoded_test, sm_encoded_test, test_stats_scaled])
        else:
            # Combine features without statistics
            X_test = np.hstack([cell_encoded_test, sm_encoded_test])
    
    # Return processed data
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'y_train_scaled': y_train_scaled,
        'X_test': X_test,
        'gene_columns': gene_columns,
        'cell_encoder': cell_encoder,
        'target_scaler': target_scaler,
        'sm_names': sm_names
    }
    
    if add_statistics:
        result['stats_scaler'] = stats_scaler
    
    if test_df is not None and 'id' in test_df.columns:
        result['test_ids'] = test_df['id'].values
    
    return result

# Function to train models with early stopping
def train_model(model, X_train, y_train, val_split=0.1, epochs=100, batch_size=64, patience=15, verbose=1):
    """
    Train a PyTorch model with early stopping
    
    Parameters:
    model (nn.Module): PyTorch model
    X_train (np.ndarray): Training features
    y_train (np.ndarray): Training targets
    val_split (float): Validation split ratio
    epochs (int): Maximum number of epochs
    batch_size (int): Batch size
    patience (int): Patience for early stopping
    verbose (int): Verbosity level
    
    Returns:
    nn.Module: Trained model
    """
    # Split data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_split, dtype=torch.float32),
        torch.tensor(y_train_split, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        if verbose > 0 and (epoch + 1) % verbose == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            if verbose > 0:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    
    return model

# Function to create and train an ensemble
def train_ensemble(X_train, y_train, output_dim, ensembles=3, verbose=1):
    """
    Create and train an ensemble of models
    
    Parameters:
    X_train (np.ndarray): Training features
    y_train (np.ndarray): Training targets
    output_dim (int): Output dimension
    ensembles (int): Number of models in the ensemble
    verbose (int): Verbosity level
    
    Returns:
    list: List of trained models
    """
    print(f"Training ensemble of {ensembles} models...")
    
    input_dim = X_train.shape[1]
    models = []
    
    # Train BiLSTM models
    for i in range(ensembles // 3 + 1):
        if len(models) >= ensembles:
            break
            
        print(f"\nTraining BiLSTM model {i+1}...")
        bilstm = AttentionBiLSTM(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=output_dim,
            num_layers=2,
            dropout=0.3
        ).to(device)
        
        trained_bilstm = train_model(
            model=bilstm,
            X_train=X_train,
            y_train=y_train,
            val_split=0.1,
            epochs=150,
            batch_size=64,
            patience=15,
            verbose=verbose
        )
        
        models.append(('bilstm', trained_bilstm))
    
    # Train CNN models
    for i in range(ensembles // 3 + 1):
        if len(models) >= ensembles:
            break
            
        print(f"\nTraining CNN model {i+1}...")
        cnn = ResidualCNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=128,
            dropout=0.3
        ).to(device)
        
        trained_cnn = train_model(
            model=cnn,
            X_train=X_train,
            y_train=y_train,
            val_split=0.1,
            epochs=150,
            batch_size=64,
            patience=15,
            verbose=verbose
        )
        
        models.append(('cnn', trained_cnn))
    
    # Train GBM models
    for i in range(ensembles // 3 + 1):
        if len(models) >= ensembles:
            break
            
        print(f"\nTraining GBM model {i+1}...")
        gbm = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42 + i
        )
        
        gbm.fit(X_train, y_train)
        models.append(('gbm', gbm))
    
    return models[:ensembles]

# Function to get ensemble predictions
def get_ensemble_predictions(models, X, scaler):
    """
    Get predictions from an ensemble of models
    
    Parameters:
    models (list): List of trained models
    X (np.ndarray): Input features
    scaler (StandardScaler): Scaler for target values
    
    Returns:
    np.ndarray: Ensemble predictions
    """
    all_preds = []
    
    for model_type, model in models:
        if model_type in ['bilstm', 'cnn']:
            # PyTorch model
            model.eval()
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
            
            preds = []
            with torch.no_grad():
                for inputs in dataloader:
                    inputs = inputs[0].to(device)
                    outputs = model(inputs)
                    preds.append(outputs.cpu().numpy())
            
            preds = np.vstack(preds)
        else:
            # Scikit-learn model
            preds = model.predict(X)
        
        all_preds.append(preds)
    
    # Average predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    
    # Inverse transform
    if scaler is not None:
        ensemble_preds = scaler.inverse_transform(ensemble_preds)
    
    return ensemble_preds

# Main function to train and predict
def train_and_predict(train_path, test_path, output_path='submission.csv', eval_mode=False):
    """
    Train an ensemble and generate predictions
    
    Parameters:
    train_path (str): Path to training data
    test_path (str): Path to test data
    output_path (str): Path to save predictions
    eval_mode (bool): Whether to evaluate MRRMSE on validation data
    
    Returns:
    tuple: (models, gene_columns, mrrmse_score) if eval_mode=True, otherwise (models, gene_columns)
    """
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
    
    # 2. Preprocess data
    data = preprocess_data(train_df, test_df, add_statistics=True)
    
    X_train = data['X_train']
    y_train_scaled = data['y_train_scaled']
    X_test = data['X_test']
    gene_columns = data['gene_columns']
    target_scaler = data['target_scaler']
    
    # 3. Train ensemble
    models = train_ensemble(
        X_train=X_train,
        y_train=y_train_scaled,
        output_dim=y_train_scaled.shape[1],
        ensembles=6,  # Use 6 models in the ensemble
        verbose=5     # Print every 5 epochs
    )
    
    # 4. Generate predictions
    print("\nGenerating predictions...")
    predictions = get_ensemble_predictions(models, X_test, target_scaler)
    
    # 5. Save predictions
    submission_df = pd.DataFrame(
        predictions,
        columns=gene_columns
    )
    
    # Add IDs 
    if 'test_ids' in data:
        submission_df.insert(0, 'id', data['test_ids'])
    
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # 6. Evaluate MRRMSE
    mrrmse_score = None
    if eval_mode:
        # Create a validation set
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, data['y_train'], test_size=0.1, random_state=42
        )
        
        # Train a small ensemble for validation
        val_models = train_ensemble(
            X_train=X_train_split,
            y_train=target_scaler.transform(y_train_split),
            output_dim=y_train_scaled.shape[1],
            ensembles=3,  # Smaller ensemble for validation
            verbose=10
        )
        
        # Generate validation predictions
        val_preds = get_ensemble_predictions(val_models, X_val, target_scaler)
        
        # Calculate MRRMSE
        mrrmse_score = mrrmse(y_val, val_preds)
        print(f"\nValidation MRRMSE: {mrrmse_score:.6f}")
    
    return (models, gene_columns, mrrmse_score) if eval_mode else (models, gene_columns)

if __name__ == "__main__":
    train_path = os.path.join("dataset", "de_train_split.parquet")
    test_path = os.path.join("dataset", "de_test_split.parquet")
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Train and generate predictions with MRRMSE evaluation
    models, gene_columns, mrrmse_score = train_and_predict(
        train_path=train_path,
        test_path=test_path,
        output_path='results/ensemble_submission.csv',
        eval_mode=True
    )
    
    print(f"Final validation MRRMSE: {mrrmse_score:.6f}")
    print("Done! Predictions saved to results/ensemble_submission.csv")
