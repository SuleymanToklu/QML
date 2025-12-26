"""
VQC Hyperparameter Tuning
===========================
This script tests different VQC configurations (feature maps, ansatz, optimizers)
and compares their performance.

Usage:
    python vqc_hyperparameter_tuning.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2, PauliFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA, SPSA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from src.data_loader import DataLoader


def create_feature_map(n_features, map_type='ZZFeatureMap', reps=2):
    """Create quantum feature map."""
    if map_type == 'ZZFeatureMap':
        return ZZFeatureMap(feature_dimension=n_features, reps=reps)
    elif map_type == 'PauliFeatureMap':
        return PauliFeatureMap(feature_dimension=n_features, reps=reps)
    else:
        raise ValueError(f"Unknown feature map type: {map_type}")


def create_ansatz(n_qubits, ansatz_type='RealAmplitudes', reps=3):
    """Create variational ansatz."""
    if ansatz_type == 'RealAmplitudes':
        return RealAmplitudes(num_qubits=n_qubits, reps=reps)
    elif ansatz_type == 'EfficientSU2':
        return EfficientSU2(num_qubits=n_qubits, reps=reps)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")


def create_optimizer(optimizer_type='COBYLA', maxiter=100):
    """Create optimizer."""
    if optimizer_type == 'COBYLA':
        return COBYLA(maxiter=maxiter)
    elif optimizer_type == 'SPSA':
        return SPSA(maxiter=maxiter)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def plot_confusion_matrix(y_true, y_pred, dataset_name, config_name, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {dataset_name}\n{config_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for VQC hyperparameter tuning."""
    
    loader = DataLoader()
    datasets = loader.list_datasets()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results')
    figures_dir = os.path.join(results_dir, 'figures')
    tables_dir = os.path.join(results_dir, 'tables')
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    # Different configurations to test
    configurations = [
        {'feature_map': 'ZZFeatureMap', 'feature_reps': 2, 'ansatz': 'RealAmplitudes', 'ansatz_reps': 3, 'optimizer': 'COBYLA', 'maxiter': 50},
        {'feature_map': 'ZZFeatureMap', 'feature_reps': 3, 'ansatz': 'RealAmplitudes', 'ansatz_reps': 3, 'optimizer': 'COBYLA', 'maxiter': 50},
        {'feature_map': 'PauliFeatureMap', 'feature_reps': 2, 'ansatz': 'RealAmplitudes', 'ansatz_reps': 3, 'optimizer': 'COBYLA', 'maxiter': 50},
        {'feature_map': 'ZZFeatureMap', 'feature_reps': 2, 'ansatz': 'EfficientSU2', 'ansatz_reps': 2, 'optimizer': 'COBYLA', 'maxiter': 50},
        {'feature_map': 'ZZFeatureMap', 'feature_reps': 2, 'ansatz': 'RealAmplitudes', 'ansatz_reps': 3, 'optimizer': 'SPSA', 'maxiter': 50},
    ]
    
    all_results = []
    
    print("=" * 80)
    print("VQC Hyperparameter Tuning")
    print("=" * 80)
    print(f"Testing {len(configurations)} different configurations\n")
    
    # Test on smaller datasets for speed
    test_datasets = ['iris', 'breast_cancer']
    
    for dataset_name in test_datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        qubit_configs = loader.list_qubit_configs(dataset_name)
        
        for n_qubits in qubit_configs[:1]:  # Test with first qubit config
            print(f"\n  Testing with {n_qubits} qubits...")
            
            try:
                X_train, X_test, y_train, y_test = loader.load_dataset(dataset_name, n_qubits=n_qubits)
                
                # Use smaller subset for faster training
                if len(X_train) > 100:
                    indices = np.random.choice(len(X_train), 100, replace=False)
                    X_train_subset = X_train[indices]
                    y_train_subset = y_train[indices]
                else:
                    X_train_subset = X_train
                    y_train_subset = y_train
                
                for i, config in enumerate(configurations):
                    config_name = f"{config['feature_map']}_{config['ansatz']}_{config['optimizer']}"
                    print(f"\n    Configuration {i+1}/{len(configurations)}: {config_name}")
                    
                    try:
                        # Create circuits
                        feature_map = create_feature_map(
                            n_qubits, 
                            config['feature_map'], 
                            config['feature_reps']
                        )
                        ansatz = create_ansatz(
                            n_qubits, 
                            config['ansatz'], 
                            config['ansatz_reps']
                        )
                        optimizer = create_optimizer(
                            config['optimizer'], 
                            config['maxiter']
                        )
                        
                        # Create and train VQC
                        vqc = VQC(
                            feature_map=feature_map,
                            ansatz=ansatz,
                            optimizer=optimizer,
                            sampler=None
                        )
                        
                        print(f"      Training...")
                        start_time = time.time()
                        vqc.fit(X_train_subset, y_train_subset)
                        training_time = time.time() - start_time
                        
                        # Evaluate
                        y_pred = vqc.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        # Store results
                        result = {
                            'dataset': dataset_name,
                            'n_qubits': n_qubits,
                            'feature_map': config['feature_map'],
                            'feature_reps': config['feature_reps'],
                            'ansatz': config['ansatz'],
                            'ansatz_reps': config['ansatz_reps'],
                            'optimizer': config['optimizer'],
                            'maxiter': config['maxiter'],
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'training_time': training_time
                        }
                        all_results.append(result)
                        
                        print(f"      Accuracy: {accuracy*100:.2f}%")
                        print(f"      Training Time: {training_time:.2f}s")
                        
                    except Exception as e:
                        print(f"      ERROR: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(results_dir, 'vqc_hyperparameter_tuning_results.csv')
        results_df.to_csv(results_path, index=False)
        
        table_path = os.path.join(tables_dir, 'vqc_hyperparameter_comparison.csv')
        results_df.to_csv(table_path, index=False)
        
        print(f"\n{'='*80}")
        print("VQC HYPERPARAMETER TUNING SUMMARY")
        print(f"{'='*80}")
        print(results_df.to_string(index=False))
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

