"""
Data Loader Module for Quantum Machine Learning Experiments

This module provides a unified interface for loading preprocessed datasets
for both classical SVM (baseline) and quantum ML models (VQC/QSVM).

KOD TASARIM NOTLARI (Neden Bu Yapı?):
- Neden Class-based?: Nesne yönelimli, state tutar (data_dir)
- Neden Bu Method'lar?: list_datasets(), load_dataset() - açık ve anlaşılır API
- Fallback Mekanizması: Eski dosya formatlarını da destekler (legacy compatibility)
- Hata Yönetimi: Dosya bulunamazsa açıklayıcı hata mesajı
- Avantaj: Tüm script'lerde aynı kod, kod tekrarı yok, bakım kolay

All datasets are preprocessed with:
- Missing value handling
- Feature scaling (StandardScaler or MinMaxScaler)
- PCA dimensionality reduction (for 2, 4, or 8 qubits)
- Train/test split (70/30) with fixed random state
"""

import os
import numpy as np
from typing import Tuple, Optional, List


class DataLoader:
    """
    Unified data loader for QML experiments.
    
    Loads preprocessed datasets from the processed directory and provides
    them in a format compatible with both classical and quantum ML models.
    """
    
    def __init__(self, data_dir: str = "1_Data/processed"):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_dir : str
            Path to the processed data directory
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                "Please run the data preparation notebook first."
            )
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
        --------
        List[str]
            List of available dataset names
        """
        files = os.listdir(self.data_dir)
        datasets = set()
        for f in files:
            if '_X_train' in f:
                dataset_name = f.split('_X_train')[0]
                datasets.add(dataset_name)
        return sorted(list(datasets))
    
    def list_qubit_configs(self, dataset_name: str) -> List[int]:
        """
        List available qubit configurations for a dataset.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        List[int]
            List of available qubit counts (e.g., [2, 4, 8])
        """
        files = os.listdir(self.data_dir)
        qubit_configs = []
        for f in files:
            if f.startswith(dataset_name) and '_X_train' in f:
                # Extract qubit count from filename
                # Format: dataset_X_train_Nqubits.npy
                parts = f.split('_X_train')
                if len(parts) > 1:
                    qubit_part = parts[1].replace('.npy', '')
                    if qubit_part.startswith('_') and 'qubits' in qubit_part:
                        qubit_count = int(qubit_part.split('qubits')[0].replace('_', ''))
                        qubit_configs.append(qubit_count)
                    elif qubit_part == '':
                        # Legacy format without qubit suffix
                        qubit_configs.append(4)  # Default assumption
        return sorted(list(set(qubit_configs)))
    
    def load_dataset(self, 
                    dataset_name: str, 
                    n_qubits: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray]:
        """
        Load a preprocessed dataset.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset (e.g., 'mnist', 'iris', 'breast_cancer')
        n_qubits : int, optional
            Number of qubits (PCA components). If None, uses the first available.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (X_train, X_test, y_train, y_test) arrays
        """
        # Determine qubit configuration
        if n_qubits is None:
            available_configs = self.list_qubit_configs(dataset_name)
            if not available_configs:
                raise ValueError(f"No qubit configurations found for {dataset_name}")
            n_qubits = available_configs[0]
            print(f"Using {n_qubits} qubits (first available configuration)")
        
        # Construct file paths
        qubit_suffix = f"_{n_qubits}qubits"
        base_path = os.path.join(self.data_dir, f"{dataset_name}")
        
        # Try with qubit suffix first
        train_path = f"{base_path}_X_train{qubit_suffix}.npy"
        test_path = f"{base_path}_X_test{qubit_suffix}.npy"
        y_train_path = f"{base_path}_y_train{qubit_suffix}.npy"
        y_test_path = f"{base_path}_y_test{qubit_suffix}.npy"
        
        # Fallback to legacy format if qubit suffix files don't exist
        if not os.path.exists(train_path):
            train_path = f"{base_path}_X_train.npy"
            test_path = f"{base_path}_X_test.npy"
            y_train_path = f"{base_path}_y_train.npy"
            y_test_path = f"{base_path}_y_test.npy"
        
        # Load arrays
        try:
            X_train = np.load(train_path)
            X_test = np.load(test_path)
            y_train = np.load(y_train_path)
            y_test = np.load(y_test_path)
        except FileNotFoundError as e:
            available = self.list_qubit_configs(dataset_name)
            raise FileNotFoundError(
                f"Dataset files not found for {dataset_name} with {n_qubits} qubits.\n"
                f"Available configurations: {available}\n"
                f"Error: {e}"
            )
        
        # Ensure arrays are in correct format
        X_train = np.asarray(X_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.int64)
        y_test = np.asarray(y_test, dtype=np.int64)
        
        return X_train, X_test, y_train, y_test
    
    def get_dataset_info(self, dataset_name: str) -> dict:
        """
        Get information about a dataset.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        dict
            Dictionary with dataset information
        """
        try:
            X_train, X_test, y_train, y_test = self.load_dataset(dataset_name)
            
            info = {
                'name': dataset_name,
                'n_features': X_train.shape[1],
                'n_train_samples': X_train.shape[0],
                'n_test_samples': X_test.shape[0],
                'n_classes': len(np.unique(np.concatenate([y_train, y_test]))),
                'train_class_distribution': dict(zip(*np.unique(y_train, return_counts=True))),
                'test_class_distribution': dict(zip(*np.unique(y_test, return_counts=True))),
                'available_qubit_configs': self.list_qubit_configs(dataset_name),
                'feature_range': (X_train.min(), X_train.max())
            }
            return info
        except Exception as e:
            return {'name': dataset_name, 'error': str(e)}


def load_data(dataset_name: str, 
              n_qubits: Optional[int] = None,
              data_dir: str = "1_Data/processed") -> Tuple[np.ndarray, np.ndarray, 
                                                            np.ndarray, np.ndarray]:
    """
    Convenience function to load a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load
    n_qubits : int, optional
        Number of qubits (PCA components). If None, uses first available.
    data_dir : str
        Path to processed data directory
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, X_test, y_train, y_test) arrays
    """
    loader = DataLoader(data_dir)
    return loader.load_dataset(dataset_name, n_qubits)


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = DataLoader()
    
    # List available datasets
    print("Available datasets:")
    datasets = loader.list_datasets()
    for ds in datasets:
        print(f"  - {ds}")
    
    # Get info for a dataset
    if datasets:
        print(f"\nDataset info for '{datasets[0]}':")
        info = loader.get_dataset_info(datasets[0])
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Load dataset
        print(f"\nLoading '{datasets[0]}' dataset...")
        X_train, X_test, y_train, y_test = loader.load_dataset(datasets[0])
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")

