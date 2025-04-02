import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.lstm_model import LSTMModel
from utils.dataset import CellDataset
from utils.helpers import set_seed, get_device
from utils.preprocessing import (
    encode_cell_types,
    encode_small_molecules,
    get_sm_names,
    get_gene_columns
)
from utils.metrics import mrrmse

def train_and_predict(train_path, test_path, output_path='submission.csv', eval_mode=False):
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    metadata_cols = ['cell_type', 'sm_name']
    if 'id' in train_df.columns:
        metadata_cols.append('id')

    cell_types = train_df['cell_type'].unique()
    sm_names = get_sm_names(train_df['sm_name'])
    gene_columns = get_gene_columns(train_df, metadata_cols)

    # Encode features
    cell_encoder = encode_cell_types(cell_types)
    cell_encoded_train = cell_encoder.transform(train_df[['cell_type']])
    cell_encoded_test = cell_encoder.transform(test_df[['cell_type']])
    sm_encoded_train = encode_small_molecules(train_df['sm_name'], sm_names)
    sm_encoded_test = encode_small_molecules(test_df['sm_name'], sm_names)

    X_train = np.hstack([cell_encoded_train, sm_encoded_train])
    X_test = np.hstack([cell_encoded_test, sm_encoded_test])
    y_train = train_df[gene_columns].astype(float).values

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_scaled, test_size=0.1, random_state=42
    )

    y_train_original = y_train.copy() if eval_mode else None
    train_indices = np.arange(len(X_train))
    _, val_indices = train_test_split(train_indices, test_size=0.1, random_state=42)

    # Datasets & loaders
    batch_size = 64
    train_loader = DataLoader(CellDataset(X_train_split, y_train_split), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CellDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CellDataset(X_test), batch_size=batch_size, shuffle=False)

    model = LSTMModel(X_train.shape[1], 128, y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    epochs = 50
    patience = 10
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    print("Training model...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = criterion(model(inputs), targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)

    # Predict test set
    print("Generating predictions...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    predictions = np.vstack(predictions)
    predictions = scaler.inverse_transform(predictions)

    submission_df = pd.DataFrame(predictions, columns=gene_columns)
    if 'id' in test_df.columns:
        submission_df.insert(0, 'id', test_df['id'])
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    if eval_mode:
        y_val_original = y_train_original[val_indices]
        val_dataset = CellDataset(X_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_predictions = []
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_predictions.append(outputs.cpu().numpy())
        val_predictions = np.vstack(val_predictions)
        val_predictions = scaler.inverse_transform(val_predictions)
        score = mrrmse(y_val_original, val_predictions)
        print(f"Validation MRRMSE: {score:.6f}")
        return model, gene_columns, score

    return model, gene_columns
