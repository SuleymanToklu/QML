# Data Pipeline Documentation

This directory contains the data preparation pipeline for the Quantum Machine Learning thesis project.

## Directory Structure

```
1_Data/
├── raw/              # Raw datasets in CSV format
├── processed/        # Preprocessed datasets in NumPy format
└── 0_Data_Preparation.ipynb  # Main data preparation notebook
```

## Datasets

The pipeline processes 6 datasets:

1. **MNIST** - Digit recognition (28x28 pixel images)
2. **USGS Earthquake** - Seismic data for imbalanced classification analysis
3. **UCI Recgym** - Sensor/IMU data (complex sensor readings)
4. **PennyLane** - QML-native benchmark datasets
5. **Breast Cancer** - Medical diagnosis dataset
6. **Iris** - Baseline dataset for algorithmic testing

## Preprocessing Pipeline

Each dataset undergoes the following preprocessing steps:

1. **Missing Value Handling**: Missing values are filled with column means
2. **Feature Scaling**: 
   - StandardScaler (zero mean, unit variance) OR
   - MinMaxScaler (scaled to [-1, 1] or [0, 1] range)
3. **Dimensionality Reduction (PCA)**: 
   - Reduced to 2, 4, or 8 components (qubits)
   - Ensures compatibility with quantum circuit constraints
4. **Data Splitting**: 
   - 70% training, 30% test
   - Fixed random state (42) for reproducibility
   - Stratified splitting to maintain class distribution

## Output Format

Processed datasets are saved as NumPy arrays:

- `{dataset}_X_train_{N}qubits.npy` - Training features
- `{dataset}_X_test_{N}qubits.npy` - Test features
- `{dataset}_y_train_{N}qubits.npy` - Training labels
- `{dataset}_y_test_{N}qubits.npy` - Test labels

Where `N` is the number of qubits (2, 4, or 8).

## Usage

### Running the Data Preparation

Execute the data preparation script:

```bash
python 1_Data/data_preparation.py
```

### Loading Data in Experiments

Use the `DataLoader` class from `src/data_loader.py`:

```python
from src.data_loader import DataLoader

# Initialize loader
loader = DataLoader()

# List available datasets
datasets = loader.list_datasets()

# Load a dataset
X_train, X_test, y_train, y_test = loader.load_dataset('iris', n_qubits=2)

# Get dataset information
info = loader.get_dataset_info('iris')
```

Or use the convenience function:

```python
from src.data_loader import load_data

X_train, X_test, y_train, y_test = load_data('breast_cancer', n_qubits=4)
```

## Reproducibility

All random operations use a fixed seed (42) to ensure reproducible results:
- NumPy random seed: 42
- Train/test split random state: 42
- PCA random state: 42

## Notes

- Raw data files are excluded from git (see `.gitignore`)
- Processed data files should be regenerated using the notebook
- The pipeline is designed to work with both classical SVM and quantum ML models (VQC/QSVM)

