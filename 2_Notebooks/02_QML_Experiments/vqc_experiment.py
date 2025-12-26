"""
Variational Quantum Classifier (VQC) Experiments
=================================================
This script implements VQC experiments on all 6 datasets with different
qubit configurations and circuit architectures.

KOD YAPISI NOTLARI:
- VQC 3 bileşenden oluşur: Feature Map + Ansatz + Optimizer
- Feature Map: Klasik veriyi kuantum durumuna kodlar (ZZFeatureMap, PauliFeatureMap)
- Ansatz: Öğrenilebilir parametreli devre (RealAmplitudes, EfficientSU2)
- Optimizer: Parametreleri optimize eder (COBYLA, SPSA)
- Neden bu yapı?: Qiskit'in standart VQC implementasyonu, modüler ve esnek

VQC uses a quantum feature map to encode classical data and a variational
ansatz circuit for classification.

Usage:
    python vqc_experiment.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for script execution
import matplotlib.pyplot as plt
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2, PauliFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA, SPSA
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from src.data_loader import DataLoader


def create_feature_map(n_features, map_type='ZZFeatureMap', reps=2):
    """Create quantum feature map for data encoding."""
    if map_type == 'ZZFeatureMap':
        return ZZFeatureMap(feature_dimension=n_features, reps=reps)
    elif map_type == 'PauliFeatureMap':
        return PauliFeatureMap(feature_dimension=n_features, reps=reps)
    else:
        raise ValueError(f"Unknown feature map type: {map_type}")


def create_ansatz(n_qubits, ansatz_type='RealAmplitudes', reps=3):
    """Create variational ansatz circuit."""
    if ansatz_type == 'RealAmplitudes':
        return RealAmplitudes(num_qubits=n_qubits, reps=reps)
    elif ansatz_type == 'EfficientSU2':
        return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement='linear')
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")


def create_optimizer(optimizer_type='COBYLA', maxiter=100):
    """Create optimizer for VQC training."""
    if optimizer_type == 'COBYLA':
        return COBYLA(maxiter=maxiter)
    elif optimizer_type == 'SPSA':
        return SPSA(maxiter=maxiter)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def main():
    """Main function to run VQC experiments."""
    
    # Initialize data loader
    loader = DataLoader()
    
    # List available datasets
    datasets = loader.list_datasets()
    print("Available datasets:", datasets)
    print(f"Total datasets: {len(datasets)}\n")
    
    # VQC Configuration
    vqc_config = {
        'feature_map_type': 'ZZFeatureMap',  # Options: 'ZZFeatureMap', 'PauliFeatureMap'
        'feature_map_reps': 2,
        'ansatz_type': 'RealAmplitudes',  # Options: 'RealAmplitudes', 'EfficientSU2'
        'ansatz_reps': 3,
        'optimizer': 'COBYLA',  # Options: 'COBYLA', 'SPSA'
        'maxiter': 100
    }
    
    print("VQC Configuration:")
    for key, value in vqc_config.items():
        print(f"  {key}: {value}")
    
    # Storage for results
    results = []
    
    print("\n" + "=" * 80)
    print("VQC Experiments - All Datasets")
    print("=" * 80)
    
    # Run experiments for all datasets and qubit configurations
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Get available qubit configurations
        qubit_configs = loader.list_qubit_configs(dataset_name)
        print(f"Available qubit configurations: {qubit_configs}\n")
        
        for n_qubits in qubit_configs:
            print(f"  Testing with {n_qubits} qubits...")
            
            try:
                # Load dataset
                X_train, X_test, y_train, y_test = loader.load_dataset(dataset_name, n_qubits=n_qubits)
                
                # Create quantum circuits
                feature_map = create_feature_map(
                    n_qubits, 
                    vqc_config['feature_map_type'], 
                    vqc_config['feature_map_reps']
                )
                ansatz = create_ansatz(
                    n_qubits, 
                    vqc_config['ansatz_type'], 
                    vqc_config['ansatz_reps']
                )
                optimizer = create_optimizer(
                    vqc_config['optimizer'], 
                    vqc_config['maxiter']
                )
                
                # Create and train VQC
                vqc = VQC(
                    feature_map=feature_map,
                    ansatz=ansatz,
                    optimizer=optimizer,
                    sampler=None  # Use default Aer simulator
                )
                
                # Training
                print(f"    Training VQC...")
                start_time = time.time()
                vqc.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluation
                y_pred = vqc.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Store results
                result = {
                    'dataset': dataset_name,
                    'n_qubits': n_qubits,
                    'n_features': X_train.shape[1],
                    'n_train': X_train.shape[0],
                    'n_test': X_test.shape[0],
                    'n_classes': len(np.unique(y_test)),
                    'feature_map': vqc_config['feature_map_type'],
                    'ansatz': vqc_config['ansatz_type'],
                    'optimizer': vqc_config['optimizer'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': training_time
                }
                results.append(result)
                
                # Print results
                print(f"    Accuracy: {accuracy*100:.2f}%")
                print(f"    Precision: {precision*100:.2f}%")
                print(f"    Recall: {recall*100:.2f}%")
                print(f"    F1-Score: {f1*100:.2f}%")
                print(f"    Training Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*80}")
    print("All VQC experiments completed!")
    print(f"{'='*80}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Display summary table
    if len(results_df) > 0:
        print("\n" + "="*80)
        print("SUMMARY RESULTS - VQC Experiments")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Save results to CSV
        results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, 'vqc_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
    else:
        print("No results to display. Please run experiments first.")
    
    # Visualization: VQC performance comparison
    if len(results_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy by dataset
        ax1 = axes[0, 0]
        pivot_acc = results_df.pivot(index='dataset', columns='n_qubits', values='accuracy')
        pivot_acc.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('VQC Accuracy by Dataset and Qubit Configuration', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Dataset', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.legend(title='Qubits', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Training time comparison
        ax2 = axes[0, 1]
        pivot_time = results_df.pivot(index='dataset', columns='n_qubits', values='training_time')
        pivot_time.plot(kind='bar', ax=ax2, width=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('VQC Training Time by Dataset and Qubit Configuration', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Dataset', fontsize=10)
        ax2.set_ylabel('Training Time (seconds)', fontsize=10)
        ax2.legend(title='Qubits', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. F1-Score comparison
        ax3 = axes[1, 0]
        pivot_f1 = results_df.pivot(index='dataset', columns='n_qubits', values='f1_score')
        pivot_f1.plot(kind='bar', ax=ax3, width=0.8, color=['#96CEB4', '#FFEAA7', '#DDA15E'])
        ax3.set_title('VQC F1-Score by Dataset and Qubit Configuration', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Dataset', fontsize=10)
        ax3.set_ylabel('F1-Score', fontsize=10)
        ax3.legend(title='Qubits', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Average metrics by qubit configuration
        ax4 = axes[1, 1]
        avg_by_qubits = results_df.groupby('n_qubits')[['accuracy', 'precision', 'recall', 'f1_score']].mean()
        avg_by_qubits.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Average VQC Metrics by Qubit Configuration', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Number of Qubits', fontsize=10)
        ax4.set_ylabel('Score', fontsize=10)
        ax4.legend(['Accuracy', 'Precision', 'Recall', 'F1-Score'], bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, 'vqc_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {fig_path}")
        plt.close()  # Close figure to free memory
    else:
        print("No results to visualize.")
    
    # Detailed classification reports for each configuration
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*80)
    
    for dataset_name in datasets:
        qubit_configs = loader.list_qubit_configs(dataset_name)
        
        for n_qubits in qubit_configs:
            try:
                X_train, X_test, y_train, y_test = loader.load_dataset(dataset_name, n_qubits=n_qubits)
                
                # Create circuits
                feature_map = create_feature_map(n_qubits, vqc_config['feature_map_type'], vqc_config['feature_map_reps'])
                ansatz = create_ansatz(n_qubits, vqc_config['ansatz_type'], vqc_config['ansatz_reps'])
                optimizer = create_optimizer(vqc_config['optimizer'], vqc_config['maxiter'])
                
                # Train VQC
                vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=None)
                vqc.fit(X_train, y_train)
                
                # Predict and report
                y_pred = vqc.predict(X_test)
                
                print(f"\n{'='*80}")
                print(f"Dataset: {dataset_name.upper()} | Qubits: {n_qubits}")
                print(f"{'='*80}")
                print(classification_report(y_test, y_pred, zero_division=0))
                
            except Exception as e:
                print(f"Error for {dataset_name} with {n_qubits} qubits: {e}")
                continue


if __name__ == "__main__":
    main()

