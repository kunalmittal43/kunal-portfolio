"""
Model Training Utilities
Provides classes and functions for training neural networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Callable


class Trainer:
    """
    A flexible trainer class for PyTorch models

    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer for training
        device: Device to train on (cpu or cuda)
        scheduler: Optional learning rate scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        scheduler: Optional[object] = None
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return {'loss': epoch_loss, 'accuracy': epoch_acc}

    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss / (pbar.n + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return {'loss': epoch_loss, 'accuracy': epoch_acc}

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_best_model: bool = True,
        model_path: str = 'best_model.pth'
    ):
        """
        Train the model for multiple epochs

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait for improvement
            save_best_model: Whether to save the best model
            model_path: Path to save the best model
        """
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nTraining on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 70)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])

            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")

            # Save best model
            if save_best_model and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("\n" + "=" * 70)
        print("Training completed!")

    def predict(self, data_loader):
        """Make predictions on a dataset"""
        self.model.eval()
        predictions = []
        targets_list = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                predictions.extend(predicted.cpu().numpy())
                targets_list.extend(targets.numpy())

        return np.array(predictions), np.array(targets_list)
