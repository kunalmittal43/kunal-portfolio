# Kunal's AI Development Portfolio

A comprehensive AI/ML development project with learning resources, best practices, and example implementations.

## Project Structure

```
kunal-portfolio/
├── src/                      # Source code
│   ├── models/              # Neural network architectures
│   │   └── neural_network.py
│   ├── utils/               # Training and data utilities
│   │   ├── trainer.py
│   │   └── data_loader.py
│   └── data_processing/     # Data preprocessing modules
├── notebooks/               # Jupyter notebooks for learning
│   └── 01_getting_started.ipynb
├── examples/                # Complete working examples
│   └── classification_example.py
├── data/                    # Data storage
│   ├── raw/                 # Raw datasets
│   └── processed/           # Processed datasets
├── tests/                   # Unit tests
├── models/                  # Saved model checkpoints
└── requirements.txt         # Python dependencies
```

## Features

### Neural Network Architectures
- **SimpleClassifier**: Feedforward neural network for classification
- **CNN_Classifier**: Convolutional neural network for image classification
- **LSTMModel**: Recurrent neural network for sequence prediction

### Utilities
- **Trainer**: Comprehensive training class with:
  - Progress tracking with tqdm
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
  - Train/validation metrics
- **Data Loaders**: Custom PyTorch datasets and data loading utilities
- **Preprocessing**: Feature normalization and data splitting

## Getting Started

### 1. Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 2. Quick Start

Run the classification example:

```bash
python examples/classification_example.py
```

### 3. Interactive Learning

Launch Jupyter notebook:

```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

## Usage Examples

### Training a Model

```python
import torch
import torch.nn as nn
from src.models import SimpleClassifier
from src.utils import Trainer, create_data_loaders

# Initialize model
model = SimpleClassifier(input_size=20, hidden_size=64, num_classes=3)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create trainer
trainer = Trainer(model, criterion, optimizer, device='cuda')

# Train
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    early_stopping_patience=5
)
```

### Loading Custom Data

```python
from src.utils import load_csv_data, create_data_loaders

# Load data from CSV
X_train, X_val, y_train, y_val = load_csv_data(
    'data/raw/dataset.csv',
    target_column='label'
)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    X_train, y_train, X_val, y_val,
    batch_size=32
)
```

## Learning Resources

### Included Notebooks
1. **01_getting_started.ipynb**: Introduction to AI development
   - Data preprocessing
   - Building neural networks
   - Training and evaluation

### Key Concepts Covered
- Deep Learning fundamentals
- PyTorch basics
- Model training and validation
- Hyperparameter tuning
- Data preprocessing
- Model evaluation metrics

## Best Practices Implemented

- **Code Organization**: Modular structure with separation of concerns
- **Type Hints**: Clear function signatures
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Robust error checking
- **Reproducibility**: Random seed setting
- **Version Control**: Git integration
- **Dependencies**: Clear requirements management

## Development Tips

### Adding New Models
1. Create model class in `src/models/`
2. Inherit from `nn.Module`
3. Implement `__init__` and `forward` methods
4. Add to `__init__.py` for easy imports

### Training Best Practices
- Use batch normalization for stable training
- Implement dropout for regularization
- Monitor both train and validation metrics
- Save best model checkpoints
- Use early stopping to prevent overfitting

### Data Management
- Keep raw data separate from processed data
- Document data preprocessing steps
- Use version control for data pipelines
- Implement data validation

## Common Tasks

### Train a Classification Model
```bash
python examples/classification_example.py
```

### Run Tests
```bash
pytest tests/
```

### Check Model Architecture
```python
from src.models import CNN_Classifier
model = CNN_Classifier(num_classes=10)
print(model)
```

## Dependencies

Core libraries:
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: ML utilities
- **Matplotlib/Seaborn**: Visualization

See `requirements.txt` for complete list.

## Contributing

When adding new features:
1. Follow existing code style
2. Add docstrings to functions/classes
3. Update README if needed
4. Test your code

## Future Enhancements

- [ ] Add more model architectures (Transformers, GANs)
- [ ] Implement advanced training techniques
- [ ] Add more preprocessing utilities
- [ ] Create deployment examples
- [ ] Add model visualization tools
- [ ] Implement hyperparameter optimization
- [ ] Add more example notebooks

## Resources for Learning

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Fast.ai](https://www.fast.ai/)
- [Papers with Code](https://paperswithcode.com/)

## License

This is a personal portfolio project for learning and development.

---

**Happy Learning and Coding!**
