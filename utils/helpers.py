import numpy as np
import torch
import os

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
