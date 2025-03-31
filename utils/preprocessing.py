import numpy as np
from sklearn.preprocessing import OneHotEncoder

def encode_cell_types(cell_types):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(np.array(cell_types).reshape(-1, 1))
    return encoder

def encode_small_molecules(sm_list, sm_names):
    encoded_features = []
    for sm in sm_list:
        sm_vector = np.zeros(len(sm_names))
        for c in sm.split('|'):
            if c in sm_names:
                idx = sm_names.index(c)
                sm_vector[idx] = 1
        encoded_features.append(sm_vector)
    return np.array(encoded_features)

def get_sm_names(sm_series):
    sm_names = set()
    for sm in sm_series:
        sm_names.update(sm.split('|'))
    return sorted(list(sm_names))

def get_gene_columns(df, metadata_cols):
    numeric_cols = []
    for col in df.columns:
        if col not in metadata_cols:
            try:
                df[col].astype(float)
                numeric_cols.append(col)
            except (ValueError, TypeError):
                print(f"Skipping non-numeric column: {col}")
    return numeric_cols
