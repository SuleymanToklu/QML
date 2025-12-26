"""
Hyperparameter Tuning for Classical SVM
========================================
This script performs hyperparameter optimization for SVM using GridSearchCV
and compares different parameter configurations.

Usage:
    python hyperparameter_tuning.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data_loader import DataLoader


def plot_confusion_matrix(y_true, y_pred, dataset_name, n_qubits, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {dataset_name} ({n_qubits} qubits)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_hyperparameter_comparison(results_df, save_path):
    """Plot hyperparameter comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy by C parameter
    if 'C' in results_df.columns:
        pivot_acc = results_df.pivot_table(values='test_accuracy', index='dataset', columns='C', aggfunc='mean')
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 0], cbar_kws={'label': 'Accuracy'})
        axes[0, 0].set_title('Accuracy by C Parameter')
        axes[0, 0].set_xlabel('C Parameter')
        axes[0, 0].set_ylabel('Dataset')
    
    # Accuracy by gamma parameter
    if 'gamma' in results_df.columns:
        pivot_gamma = results_df.pivot_table(values='test_accuracy', index='dataset', columns='gamma', aggfunc='mean')
        sns.heatmap(pivot_gamma, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 1], cbar_kws={'label': 'Accuracy'})
        axes[0, 1].set_title('Accuracy by Gamma Parameter')
        axes[0, 1].set_xlabel('Gamma Parameter')
        axes[0, 1].set_ylabel('Dataset')
    
    # Training time comparison
    if 'training_time' in results_df.columns:
        time_pivot = results_df.pivot_table(values='training_time', index='dataset', columns='C', aggfunc='mean')
        sns.heatmap(time_pivot, annot=True, fmt='.2f', cmap='viridis', ax=axes[1, 0], cbar_kws={'label': 'Time (s)'})
        axes[1, 0].set_title('Training Time by C Parameter')
        axes[1, 0].set_xlabel('C Parameter')
        axes[1, 0].set_ylabel('Dataset')
    
    # F1-Score comparison
    if 'f1_score' in results_df.columns and 'C' in results_df.columns:
        f1_pivot = results_df.pivot_table(values='f1_score', index='dataset', columns='C', aggfunc='mean')
        sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1, 1], cbar_kws={'label': 'F1-Score'})
        axes[1, 1].set_title('F1-Score by C Parameter')
        axes[1, 1].set_xlabel('C Parameter')
        axes[1, 1].set_ylabel('Dataset')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for hyperparameter tuning."""
    
    # Initialize data loader
    loader = DataLoader()
    datasets = loader.list_datasets()
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results')
    figures_dir = os.path.join(results_dir, 'figures')
    tables_dir = os.path.join(results_dir, 'tables')
    metrics_dir = os.path.join(results_dir, 'metrics')
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Storage for results
    all_results = []
    
    print("=" * 80)
    print("SVM Hyperparameter Tuning")
    print("=" * 80)
    print(f"Parameter Grid: {param_grid}\n")
    
    # Test on smaller subset for faster execution
    test_datasets = ['iris', 'breast_cancer']  # Start with smaller datasets
    
    for dataset_name in test_datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        qubit_configs = loader.list_qubit_configs(dataset_name)
        
        for n_qubits in qubit_configs[:1]:  # Test with first qubit config for speed
            print(f"\n  Testing with {n_qubits} qubits...")
            
            try:
                # Load dataset
                X_train, X_test, y_train, y_test = loader.load_dataset(dataset_name, n_qubits=n_qubits)
                
                # Use smaller subset for grid search (faster)
                if len(X_train) > 500:
                    indices = np.random.choice(len(X_train), 500, replace=False)
                    X_train_subset = X_train[indices]
                    y_train_subset = y_train[indices]
                else:
                    X_train_subset = X_train
                    y_train_subset = y_train
                
                # Grid search with cross-validation
                print(f"    Running GridSearchCV (this may take a while)...")
                svm = SVC()
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                grid_search = GridSearchCV(
                    svm, param_grid, cv=cv, 
                    scoring='accuracy', n_jobs=-1, verbose=1
                )
                
                start_time = time.time()
                grid_search.fit(X_train_subset, y_train_subset)
                tuning_time = time.time() - start_time
                
                # Best parameters
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                
                print(f"    Best parameters: {best_params}")
                print(f"    Best CV score: {best_score:.4f}")
                
                # Train with best parameters on full training set
                best_svm = SVC(**best_params)
                train_start = time.time()
                best_svm.fit(X_train, y_train)
                training_time = time.time() - train_start
                
                # Evaluate
                y_pred = best_svm.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Store results
                result = {
                    'dataset': dataset_name,
                    'n_qubits': n_qubits,
                    'C': best_params['C'],
                    'gamma': str(best_params['gamma']),
                    'kernel': best_params['kernel'],
                    'best_cv_score': best_score,
                    'test_accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'tuning_time': tuning_time,
                    'training_time': training_time,
                    'total_time': tuning_time + training_time
                }
                all_results.append(result)
                
                # Save confusion matrix
                cm_path = os.path.join(figures_dir, f'confusion_matrix_{dataset_name}_{n_qubits}qubits.png')
                plot_confusion_matrix(y_test, y_pred, dataset_name, n_qubits, cm_path)
                
                print(f"    Test Accuracy: {accuracy*100:.2f}%")
                print(f"    F1-Score: {f1*100:.2f}%")
                print(f"    Tuning Time: {tuning_time:.2f}s")
                print(f"    Training Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create results DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_path = os.path.join(results_dir, 'svm_hyperparameter_tuning_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {results_path}")
        
        # Save comparison table
        table_path = os.path.join(tables_dir, 'hyperparameter_comparison.csv')
        results_df.to_csv(table_path, index=False)
        
        # Plot comparison
        comparison_path = os.path.join(figures_dir, 'hyperparameter_comparison.png')
        plot_hyperparameter_comparison(results_df, comparison_path)
        print(f"Comparison plot saved to: {comparison_path}")
        
        # Display summary
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING SUMMARY")
        print(f"{'='*80}")
        print(results_df.to_string(index=False))
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    main()

