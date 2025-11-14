"""
Data Loading and Preprocessing Utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import pandas as pd


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for flexible data loading

    Args:
        X: Input features (numpy array or tensor)
        y: Target labels (numpy array or tensor)
        transform: Optional transform to apply to the data
    """

    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        self.y = torch.LongTensor(y) if not isinstance(y, torch.Tensor) else y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def create_data_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders for training and validation

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle training data
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def normalize_features(X_train, X_val=None, X_test=None):
    """
    Normalize features using training set statistics

    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)

    Returns:
        Normalized datasets
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train_normalized = (X_train - mean) / std

    results = [X_train_normalized]

    if X_val is not None:
        X_val_normalized = (X_val - mean) / std
        results.append(X_val_normalized)

    if X_test is not None:
        X_test_normalized = (X_test - mean) / std
        results.append(X_test_normalized)

    return results if len(results) > 1 else results[0]


def load_csv_data(
    file_path: str,
    target_column: str,
    feature_columns: Optional[list] = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Load and split data from CSV file

    Args:
        file_path: Path to CSV file
        target_column: Name of target column
        feature_columns: List of feature column names (uses all if None)
        test_size: Proportion of data for validation
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_val, y_train, y_val
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(file_path)

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]

    X = df[feature_columns].values
    y = df[target_column].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, y_train, y_val
