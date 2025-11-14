"""
Complete Classification Example
Demonstrates how to train a neural network classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('../')

from src.models import SimpleClassifier
from src.utils import Trainer, create_data_loaders


def main():
    # Configuration
    RANDOM_SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("Neural Network Classification Example")
    print("=" * 70)

    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=RANDOM_SEED
    )

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=BATCH_SIZE
    )

    # Initialize model
    model = SimpleClassifier(
        input_size=20,
        hidden_size=64,
        num_classes=3,
        dropout_rate=0.3
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        scheduler=scheduler
    )

    # Train the model
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        early_stopping_patience=5,
        save_best_model=True,
        model_path='models/best_classifier.pth'
    )

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    test_loader, _ = create_data_loaders(
        X_test, y_test, X_test, y_test,
        batch_size=BATCH_SIZE
    )

    test_metrics = trainer.validate(test_loader[0] if isinstance(test_loader, tuple) else test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")


if __name__ == '__main__':
    main()
