"""
SVM Parameter Comparison - Multiple Configurations
===================================================
This script tests SVM with different parameter configurations and compares results.

KOD YAPISI NOTLARI:
- 12 farklı parametre konfigürasyonu test edilir
- Her konfigürasyon için confusion matrix ve ROC curve oluşturulur
- Sonuçlar CSV'ye kaydedilir ve görselleştirilir
- Neden 12 konfigürasyon?: C (4 değer) × gamma (3 değer) × kernel (3 değer) kombinasyonlarından önemli olanlar seçildi

Usage:
    python svm_parameter_comparison.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import sys
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data_loader import DataLoader


def plot_roc_curve(y_test, y_pred_proba, dataset_name, n_qubits, n_classes, save_path):
    """Plot ROC curve for binary classification."""
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_name} ({n_qubits} qubits)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_parameter_comparison(results_df, save_path):
    """Plot comprehensive parameter comparison."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy by C parameter
    ax1 = fig.add_subplot(gs[0, 0])
    if 'C' in results_df.columns:
        pivot_c = results_df.pivot_table(values='accuracy', index='dataset', columns='C', aggfunc='mean')
        sns.heatmap(pivot_c, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Accuracy'})
        ax1.set_title('Accuracy by C Parameter', fontsize=11, fontweight='bold')
        ax1.set_xlabel('C Parameter')
        ax1.set_ylabel('Dataset')
    
    # 2. Accuracy by gamma parameter
    ax2 = fig.add_subplot(gs[0, 1])
    if 'gamma' in results_df.columns:
        pivot_gamma = results_df.pivot_table(values='accuracy', index='dataset', columns='gamma', aggfunc='mean')
        sns.heatmap(pivot_gamma, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Accuracy'})
        ax2.set_title('Accuracy by Gamma Parameter', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Gamma Parameter')
        ax2.set_ylabel('Dataset')
    
    # 3. Accuracy by kernel
    ax3 = fig.add_subplot(gs[0, 2])
    if 'kernel' in results_df.columns:
        pivot_kernel = results_df.pivot_table(values='accuracy', index='dataset', columns='kernel', aggfunc='mean')
        sns.heatmap(pivot_kernel, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Accuracy'})
        ax3.set_title('Accuracy by Kernel', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Kernel')
        ax3.set_ylabel('Dataset')
    
    # 4. Training time by C
    ax4 = fig.add_subplot(gs[1, 0])
    if 'C' in results_df.columns and 'training_time' in results_df.columns:
        time_pivot = results_df.pivot_table(values='training_time', index='dataset', columns='C', aggfunc='mean')
        sns.heatmap(time_pivot, annot=True, fmt='.2f', cmap='viridis', ax=ax4, cbar_kws={'label': 'Time (s)'})
        ax4.set_title('Training Time by C Parameter', fontsize=11, fontweight='bold')
        ax4.set_xlabel('C Parameter')
        ax4.set_ylabel('Dataset')
    
    # 5. F1-Score by C
    ax5 = fig.add_subplot(gs[1, 1])
    if 'C' in results_df.columns and 'f1_score' in results_df.columns:
        f1_pivot = results_df.pivot_table(values='f1_score', index='dataset', columns='C', aggfunc='mean')
        sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax5, cbar_kws={'label': 'F1-Score'})
        ax5.set_title('F1-Score by C Parameter', fontsize=11, fontweight='bold')
        ax5.set_xlabel('C Parameter')
        ax5.set_ylabel('Dataset')
    
    # 6. Precision by C
    ax6 = fig.add_subplot(gs[1, 2])
    if 'C' in results_df.columns and 'precision' in results_df.columns:
        prec_pivot = results_df.pivot_table(values='precision', index='dataset', columns='C', aggfunc='mean')
        sns.heatmap(prec_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax6, cbar_kws={'label': 'Precision'})
        ax6.set_title('Precision by C Parameter', fontsize=11, fontweight='bold')
        ax6.set_xlabel('C Parameter')
        ax6.set_ylabel('Dataset')
    
    # 7. Accuracy comparison bar chart
    ax7 = fig.add_subplot(gs[2, 0])
    if 'C' in results_df.columns:
        acc_by_c = results_df.groupby('C')['accuracy'].mean().sort_index()
        acc_by_c.plot(kind='bar', ax=ax7, color='steelblue', width=0.6)
        ax7.set_title('Average Accuracy by C Parameter', fontsize=11, fontweight='bold')
        ax7.set_xlabel('C Parameter')
        ax7.set_ylabel('Average Accuracy')
        ax7.grid(axis='y', alpha=0.3)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=0)
    
    # 8. Performance vs Time scatter
    ax8 = fig.add_subplot(gs[2, 1])
    if 'accuracy' in results_df.columns and 'training_time' in results_df.columns:
        scatter = ax8.scatter(results_df['training_time'], results_df['accuracy'], 
                             c=results_df['C'] if 'C' in results_df.columns else None,
                             s=100, alpha=0.6, cmap='viridis')
        ax8.set_xlabel('Training Time (s)')
        ax8.set_ylabel('Accuracy')
        ax8.set_title('Accuracy vs Training Time', fontsize=11, fontweight='bold')
        ax8.grid(alpha=0.3)
        if 'C' in results_df.columns:
            plt.colorbar(scatter, ax=ax8, label='C Parameter')
    
    # 9. Best configuration per dataset
    ax9 = fig.add_subplot(gs[2, 2])
    if 'dataset' in results_df.columns and 'accuracy' in results_df.columns:
        best_configs = results_df.loc[results_df.groupby('dataset')['accuracy'].idxmax()]
        best_configs = best_configs.sort_values('accuracy', ascending=True)
        y_pos = np.arange(len(best_configs))
        ax9.barh(y_pos, best_configs['accuracy'], color='green', alpha=0.7)
        ax9.set_yticks(y_pos)
        ax9.set_yticklabels(best_configs['dataset'])
        ax9.set_xlabel('Accuracy')
        ax9.set_title('Best Configuration per Dataset', fontsize=11, fontweight='bold')
        ax9.grid(axis='x', alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for parameter comparison."""
    
    loader = DataLoader()
    datasets = loader.list_datasets()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results')
    figures_dir = os.path.join(results_dir, 'figures')
    tables_dir = os.path.join(results_dir, 'tables')
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    # Different parameter configurations to test
    configurations = [
        {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'},
        {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'},  # Default
        {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'},
        {'C': 100.0, 'gamma': 'scale', 'kernel': 'rbf'},
        {'C': 1.0, 'gamma': 'auto', 'kernel': 'rbf'},
        {'C': 1.0, 'gamma': 0.001, 'kernel': 'rbf'},
        {'C': 1.0, 'gamma': 0.01, 'kernel': 'rbf'},
        {'C': 1.0, 'gamma': 0.1, 'kernel': 'rbf'},
        {'C': 1.0, 'gamma': 'scale', 'kernel': 'poly'},
        {'C': 1.0, 'gamma': 'scale', 'kernel': 'sigmoid'},
        {'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'},
        {'C': 100.0, 'gamma': 0.1, 'kernel': 'rbf'},
    ]
    
    all_results = []
    
    print("=" * 80)
    print("SVM Parameter Comparison")
    print("=" * 80)
    print(f"Testing {len(configurations)} different parameter configurations\n")
    
    # Test on selected datasets
    test_datasets = ['iris', 'breast_cancer', 'mnist']
    
    for dataset_name in test_datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        qubit_configs = loader.list_qubit_configs(dataset_name)
        
        for n_qubits in qubit_configs[:1]:  # Test with first qubit config
            print(f"\n  Testing with {n_qubits} qubits...")
            
            try:
                X_train, X_test, y_train, y_test = loader.load_dataset(dataset_name, n_qubits=n_qubits)
                n_classes = len(np.unique(y_test))
                
                for i, config in enumerate(configurations):
                    config_str = f"C={config['C']}, gamma={config['gamma']}, kernel={config['kernel']}"
                    print(f"    Config {i+1}/{len(configurations)}: {config_str}")
                    
                    try:
                        # Create and train SVM
                        svm = SVC(**config, probability=(n_classes == 2))
                        
                        start_time = time.time()
                        svm.fit(X_train, y_train)
                        training_time = time.time() - start_time
                        
                        # Predictions
                        y_pred = svm.predict(X_test)
                        
                        # Metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        # Store results
                        result = {
                            'dataset': dataset_name,
                            'n_qubits': n_qubits,
                            'C': config['C'],
                            'gamma': str(config['gamma']),
                            'kernel': config['kernel'],
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'training_time': training_time
                        }
                        all_results.append(result)
                        
                        # Save confusion matrix for best configs
                        if accuracy > 0.9 or i == 0:  # Save for high accuracy or first config
                            cm_path = os.path.join(figures_dir, 
                                f'confusion_matrix_{dataset_name}_{n_qubits}qubits_C{config["C"]}_g{str(config["gamma"])}_k{config["kernel"]}.png')
                            cm = confusion_matrix(y_test, y_pred)
                            plt.figure(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
                            plt.title(f'Confusion Matrix - {dataset_name}\n{config_str}')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            plt.tight_layout()
                            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                            plt.close()
                        
                        # ROC curve for binary classification
                        if n_classes == 2:
                            y_pred_proba = svm.predict_proba(X_test)
                            roc_path = os.path.join(figures_dir,
                                f'roc_curve_{dataset_name}_{n_qubits}qubits_C{config["C"]}_g{str(config["gamma"])}_k{config["kernel"]}.png')
                            plot_roc_curve(y_test, y_pred_proba, dataset_name, n_qubits, n_classes, roc_path)
                        
                        print(f"      Accuracy: {accuracy*100:.2f}% | Time: {training_time:.3f}s")
                        
                    except Exception as e:
                        print(f"      ERROR: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                continue
    
    # Save and visualize results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_path = os.path.join(results_dir, 'svm_parameter_comparison_results.csv')
        results_df.to_csv(results_path, index=False)
        
        table_path = os.path.join(tables_dir, 'svm_parameter_comparison.csv')
        results_df.to_csv(table_path, index=False)
        
        # Plot comprehensive comparison
        comparison_path = os.path.join(figures_dir, 'svm_parameter_comparison.png')
        plot_parameter_comparison(results_df, comparison_path)
        
        print(f"\n{'='*80}")
        print("PARAMETER COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(results_df.groupby(['C', 'gamma', 'kernel']).agg({
            'accuracy': 'mean',
            'f1_score': 'mean',
            'training_time': 'mean'
        }).round(4))
        
        print(f"\nResults saved to: {results_path}")
        print(f"Comparison plot saved to: {comparison_path}")


if __name__ == "__main__":
    main()

