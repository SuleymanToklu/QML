"""
Classical SVM Baseline Experiments
==================================
This script implements classical Support Vector Machine (SVM) as baseline
for comparison with quantum ML models (VQC and QSVM).

KOD YAPISI NOTLARI (Kod Nasıl Yazıldı):
1. Data Loader Kullanımı: Tüm datasetler için aynı API
2. Loop Yapısı: Tüm dataset ve qubit kombinasyonlarını otomatik test eder
3. Metrik Toplama: Accuracy, precision, recall, f1_score, training_time
4. Sonuç Kaydetme: CSV formatında kaydeder, görselleştirme yapar
5. Reproducibility: Random seed sabit (42), tüm sonuçlar tekrarlanabilir

Experiments are conducted on all 6 datasets with different qubit configurations.

Usage:
    python svm_baseline.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for script execution
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from src.data_loader import DataLoader


def main():
    """Main function to run SVM baseline experiments."""
    
    # Initialize data loader
    loader = DataLoader()
    
    # List available datasets
    datasets = loader.list_datasets()
    print("Available datasets:", datasets)
    print(f"\nTotal datasets: {len(datasets)}")
    
    # SVM Configuration
    # Using RBF kernel as it's commonly used and comparable to quantum kernels
    svm_config = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    }
    
    # Storage for results
    results = []
    
    print("=" * 80)
    print("Classical SVM Baseline Experiments")
    print("=" * 80)
    print(f"SVM Configuration: {svm_config}\n")
    
    # Run experiments for all datasets and qubit configurations
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Get available qubit configurations for this dataset
        qubit_configs = loader.list_qubit_configs(dataset_name)
        print(f"Available qubit configurations: {qubit_configs}\n")
        
        for n_qubits in qubit_configs:
            print(f"  Testing with {n_qubits} qubits (PCA components)...")
            
            try:
                # Load dataset
                X_train, X_test, y_train, y_test = loader.load_dataset(dataset_name, n_qubits=n_qubits)
                
                # Create and train SVM model
                model = SVC(**svm_config)
                
                # Training
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Prediction
                y_pred = model.predict(X_test)
                
                # Calculate metrics
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
    print("All experiments completed!")
    print(f"{'='*80}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Display summary table
    print("\n" + "="*80)
    print("SUMMARY RESULTS - Classical SVM Baseline")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'svm_baseline_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Visualization: Accuracy comparison across datasets and qubit configurations
    if len(results_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy by dataset
        ax1 = axes[0, 0]
        pivot_acc = results_df.pivot(index='dataset', columns='n_qubits', values='accuracy')
        pivot_acc.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('SVM Accuracy by Dataset and Qubit Configuration', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Dataset', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.legend(title='Qubits', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Training time comparison
        ax2 = axes[0, 1]
        pivot_time = results_df.pivot(index='dataset', columns='n_qubits', values='training_time')
        pivot_time.plot(kind='bar', ax=ax2, width=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('SVM Training Time by Dataset and Qubit Configuration', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Dataset', fontsize=10)
        ax2.set_ylabel('Training Time (seconds)', fontsize=10)
        ax2.legend(title='Qubits', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. F1-Score comparison
        ax3 = axes[1, 0]
        pivot_f1 = results_df.pivot(index='dataset', columns='n_qubits', values='f1_score')
        pivot_f1.plot(kind='bar', ax=ax3, width=0.8, color=['#96CEB4', '#FFEAA7', '#DDA15E'])
        ax3.set_title('SVM F1-Score by Dataset and Qubit Configuration', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Dataset', fontsize=10)
        ax3.set_ylabel('F1-Score', fontsize=10)
        ax3.legend(title='Qubits', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Average metrics by qubit configuration
        ax4 = axes[1, 1]
        avg_by_qubits = results_df.groupby('n_qubits')[['accuracy', 'precision', 'recall', 'f1_score']].mean()
        avg_by_qubits.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Average SVM Metrics by Qubit Configuration', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Number of Qubits', fontsize=10)
        ax4.set_ylabel('Score', fontsize=10)
        ax4.legend(['Accuracy', 'Precision', 'Recall', 'F1-Score'], bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, 'svm_baseline_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {fig_path}")
        plt.close()  # Close figure to free memory
    else:
        print("No results to visualize. Please run experiments first.")
    
    # Generate detailed classification reports for each configuration
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*80)
    
    for dataset_name in datasets:
        qubit_configs = loader.list_qubit_configs(dataset_name)
        
        for n_qubits in qubit_configs:
            try:
                X_train, X_test, y_train, y_test = loader.load_dataset(dataset_name, n_qubits=n_qubits)
                
                model = SVC(**svm_config)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                print(f"\n{'='*80}")
                print(f"Dataset: {dataset_name.upper()} | Qubits: {n_qubits}")
                print(f"{'='*80}")
                print(classification_report(y_test, y_pred, zero_division=0))
                
            except Exception as e:
                print(f"Error for {dataset_name} with {n_qubits} qubits: {e}")
                continue


if __name__ == "__main__":
    main()

