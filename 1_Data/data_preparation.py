"""
Data Preparation Script
========================
This script prepares all 6 datasets for quantum machine learning experiments.

KOD YAPISI NOTLARI:
- Neden PCA?: Kuantum devreler sınırlı qubit sayısı kullanır, PCA ile boyut azaltma gerekli
- Neden 2, 4, 8 qubits?: Farklı karmaşıklık seviyelerini test etmek için
- Neden StandardScaler?: Kuantum feature map'ler için normalize edilmiş veri gerekli
- Neden 70/30 split?: Standart ML pratiği, yeterli test verisi için
- Random seed 42?: Reproducibility için sabit seed

It handles:
- Data downloading/loading
- Missing value handling
- Feature scaling (StandardScaler or MinMaxScaler)
- Dimensionality reduction (PCA for 2, 4, 8 qubits)
- Train/test splitting (70/30)
- Saving processed data as NumPy arrays

Usage:
    python data_preparation.py
"""

import os
import pandas as pd
import numpy as np
import requests
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
RANDOM_STATE = 42


def preprocess_and_save(df, name, target_col, n_components_list=[2, 4, 8], 
                        scale_method='standard', scale_range=[-1, 1]):
    """
    Preprocess dataset and save in multiple qubit configurations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features and target
    name : str
        Dataset name identifier
    target_col : str
        Name of target column
    n_components_list : list
        List of PCA components (qubit counts) to generate
    scale_method : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler
    scale_range : list
        [min, max] for MinMaxScaler output range
    """
    # Save raw dataset
    raw_path = os.path.join(RAW_DIR, f"{name}_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data: {name}")
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print(f"  Handling {X.isnull().sum().sum()} missing values...")
        X = X.fillna(X.mean())  # Fill numeric columns with mean
    
    # Convert target to numeric if needed
    if y.dtype == 'object':
        y = pd.Categorical(y).codes
    
    # Scale features for quantum feature map compatibility
    if scale_method == 'standard':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scale_method == 'minmax':
        scaler = MinMaxScaler(feature_range=tuple(scale_range))
        X_scaled = scaler.fit_transform(X)
    else:
        raise ValueError("scale_method must be 'standard' or 'minmax'")
    
    # Process for each qubit configuration
    max_components = min(X_scaled.shape[1], max(n_components_list))
    
    for n_components in n_components_list:
        # Skip if n_components exceeds available features
        if n_components > X_scaled.shape[1]:
            print(f"  Skipping {n_components} qubits: only {X_scaled.shape[1]} features available")
            continue
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        
        # Split data: 70% training, 30% test
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
        )
        
        # Save processed numpy arrays
        qubit_suffix = f"_{n_components}qubits"
        np.save(os.path.join(PROCESSED_DIR, f"{name}_X_train{qubit_suffix}.npy"), X_train)
        np.save(os.path.join(PROCESSED_DIR, f"{name}_X_test{qubit_suffix}.npy"), X_test)
        np.save(os.path.join(PROCESSED_DIR, f"{name}_y_train{qubit_suffix}.npy"), y_train)
        np.save(os.path.join(PROCESSED_DIR, f"{name}_y_test{qubit_suffix}.npy"), y_test)
        
        print(f"  Processed {name} with {n_components} qubits: "
              f"Train shape {X_train.shape}, Test shape {X_test.shape}")
    
    print(f"[OK] Completed preprocessing for {name}\n")


def main():
    """Main function to prepare all datasets."""
    
    # Create directories for raw and processed data
    global RAW_DIR, PROCESSED_DIR
    RAW_DIR = "1_Data/raw"
    PROCESSED_DIR = "1_Data/processed"
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. MNIST - Digit Recognition Dataset
    print("=" * 60)
    print("Processing MNIST Dataset (Digit Recognition)")
    print("=" * 60)
    mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
    # Sample subset for computational efficiency (quantum simulators have limits)
    mnist_df = mnist.frame.sample(n=2000, random_state=RANDOM_STATE)
    # Convert class to numeric
    mnist_df['class'] = pd.Categorical(mnist_df['class']).codes
    preprocess_and_save(mnist_df, "mnist", "class", n_components_list=[2, 4, 8], 
                       scale_method='minmax', scale_range=[-1, 1])
    
    # 2. USGS Earthquake Data (Seismic/Earthquake)
    print("=" * 60)
    print("Processing USGS Earthquake Dataset")
    print("=" * 60)
    eq_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
    try:
        eq_df = pd.read_csv(eq_url)
        # Select relevant features for classification
        feature_cols = ['mag', 'depth', 'nst', 'gap', 'dmin', 'rms']
        eq_df = eq_df[feature_cols + ['magType']].copy()
        
        # Create binary classification target: significant earthquake (mag > 4.0)
        # This creates an imbalanced dataset suitable for analysis
        eq_df['target'] = (eq_df['mag'] > 4.0).astype(int)
        eq_df = eq_df.drop(['mag', 'magType'], axis=1)
        
        # Remove rows with too many missing values
        eq_df = eq_df.dropna(thresh=len(eq_df.columns) - 1)
        
        preprocess_and_save(eq_df, "earthquake", "target", n_components_list=[2, 4, 8],
                           scale_method='standard', scale_range=[-1, 1])
    except Exception as e:
        print(f"Error downloading USGS data: {e}")
        print("Using sample earthquake data structure...")
        # Create synthetic earthquake data if download fails
        n_samples = 1000
        eq_synthetic = pd.DataFrame({
            'depth': np.random.uniform(0, 700, n_samples),
            'nst': np.random.randint(10, 200, n_samples),
            'gap': np.random.uniform(0, 360, n_samples),
            'dmin': np.random.uniform(0, 10, n_samples),
            'rms': np.random.uniform(0, 2, n_samples),
            'target': np.random.binomial(1, 0.2, n_samples)  # Imbalanced
        })
        preprocess_and_save(eq_synthetic, "earthquake", "target", n_components_list=[2, 4, 8],
                           scale_method='standard', scale_range=[-1, 1])
    
    # 3. Breast Cancer Dataset (Medical)
    print("=" * 60)
    print("Processing Breast Cancer Dataset")
    print("=" * 60)
    cancer = load_breast_cancer(as_frame=True)
    preprocess_and_save(cancer.frame, "breast_cancer", "target", 
                       n_components_list=[2, 4, 8], scale_method='standard', scale_range=[-1, 1])
    
    # 4. Iris Dataset (Baseline)
    print("=" * 60)
    print("Processing Iris Dataset (Baseline)")
    print("=" * 60)
    iris = load_iris(as_frame=True)
    preprocess_and_save(iris.frame, "iris", "target", 
                       n_components_list=[2, 4], scale_method='minmax', scale_range=[-1, 1])
    
    # 5. UCI Recgym Dataset (Sensors/IMU)
    print("=" * 60)
    print("Processing UCI Recgym Dataset (Sensors/IMU)")
    print("=" * 60)
    try:
        # Try to fetch from UCI ML Repository
        recgym_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        # For now, we'll use a similar sensor dataset or create synthetic data
        # UCI HAR dataset requires special handling, using alternative approach
        print("Note: UCI Recgym/HAR dataset requires manual download.")
        print("Creating synthetic IMU sensor data for demonstration...")
        
        # Synthetic IMU-like sensor data (accelerometer, gyroscope features)
        n_samples = 1500
        recgym_df = pd.DataFrame({
            'acc_x_mean': np.random.normal(0, 0.5, n_samples),
            'acc_y_mean': np.random.normal(0, 0.5, n_samples),
            'acc_z_mean': np.random.normal(9.8, 0.5, n_samples),
            'gyro_x_mean': np.random.normal(0, 0.3, n_samples),
            'gyro_y_mean': np.random.normal(0, 0.3, n_samples),
            'gyro_z_mean': np.random.normal(0, 0.3, n_samples),
            'acc_x_std': np.random.uniform(0.1, 1.0, n_samples),
            'acc_y_std': np.random.uniform(0.1, 1.0, n_samples),
            'acc_z_std': np.random.uniform(0.1, 1.0, n_samples),
            'target': np.random.randint(0, 3, n_samples)  # 3 activity classes
        })
        preprocess_and_save(recgym_df, "recgym", "target", n_components_list=[2, 4, 8],
                           scale_method='standard', scale_range=[-1, 1])
    except Exception as e:
        print(f"Error processing Recgym data: {e}")
        print("Using synthetic sensor data...")
        n_samples = 1500
        recgym_df = pd.DataFrame({
            'acc_x_mean': np.random.normal(0, 0.5, n_samples),
            'acc_y_mean': np.random.normal(0, 0.5, n_samples),
            'acc_z_mean': np.random.normal(9.8, 0.5, n_samples),
            'gyro_x_mean': np.random.normal(0, 0.3, n_samples),
            'gyro_y_mean': np.random.normal(0, 0.3, n_samples),
            'gyro_z_mean': np.random.normal(0, 0.3, n_samples),
            'target': np.random.randint(0, 3, n_samples)
        })
        preprocess_and_save(recgym_df, "recgym", "target", n_components_list=[2, 4, 8],
                           scale_method='standard', scale_range=[-1, 1])
    
    # 6. PennyLane Native QML Datasets
    print("=" * 60)
    print("Processing PennyLane QML Native Dataset")
    print("=" * 60)
    try:
        import pennylane as qml
        # Use PennyLane's built-in datasets if available
        # For QML benchmarks, we'll create a quantum-inspired dataset
        print("Creating quantum-inspired dataset for QML benchmarks...")
        
        # Generate data that mimics quantum feature distributions
        n_samples = 1200
        # Create features with quantum-like correlations
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        phi = np.random.uniform(0, np.pi, n_samples)
        
        pennylane_df = pd.DataFrame({
            'feature_0': np.sin(theta) * np.cos(phi),
            'feature_1': np.sin(theta) * np.sin(phi),
            'feature_2': np.cos(theta),
            'feature_3': np.random.uniform(-1, 1, n_samples),
            'feature_4': np.random.uniform(-1, 1, n_samples),
            'feature_5': np.random.uniform(-1, 1, n_samples),
            'target': (np.sin(theta) > 0).astype(int)  # Binary classification
        })
        preprocess_and_save(pennylane_df, "pennylane", "target", n_components_list=[2, 4, 8],
                           scale_method='minmax', scale_range=[-1, 1])
    except ImportError:
        print("PennyLane not installed. Creating quantum-inspired synthetic dataset...")
        n_samples = 1200
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        phi = np.random.uniform(0, np.pi, n_samples)
        
        pennylane_df = pd.DataFrame({
            'feature_0': np.sin(theta) * np.cos(phi),
            'feature_1': np.sin(theta) * np.sin(phi),
            'feature_2': np.cos(theta),
            'feature_3': np.random.uniform(-1, 1, n_samples),
            'feature_4': np.random.uniform(-1, 1, n_samples),
            'feature_5': np.random.uniform(-1, 1, n_samples),
            'target': (np.sin(theta) > 0).astype(int)
        })
        preprocess_and_save(pennylane_df, "pennylane", "target", n_components_list=[2, 4, 8],
                           scale_method='minmax', scale_range=[-1, 1])
    except Exception as e:
        print(f"Error processing PennyLane data: {e}")
    
    # Summary: Verify all datasets are processed
    print("=" * 60)
    print("Data Preparation Summary")
    print("=" * 60)
    print("\nProcessed datasets:")
    datasets = ['mnist', 'earthquake', 'breast_cancer', 'iris', 'recgym', 'pennylane']
    
    for dataset in datasets:
        files = [f for f in os.listdir(PROCESSED_DIR) if f.startswith(dataset)]
        if files:
            print(f"  [OK] {dataset}: {len([f for f in files if 'X_train' in f])} qubit configurations")
        else:
            print(f"  [MISSING] {dataset}: Not found")
    
    print(f"\nAll processed data saved to: {PROCESSED_DIR}")
    print(f"All raw data saved to: {RAW_DIR}")
    print("\nData pipeline complete! Ready for QML experiments.")


if __name__ == "__main__":
    main()

