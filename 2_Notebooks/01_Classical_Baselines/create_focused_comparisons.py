"""
Create Focused Comparison Visualizations
========================================
This script creates simple, focused visualizations comparing:
1. Parameter changes (C, gamma, kernel)
2. Qubit changes (2, 4, 8 qubits)
3. Circuit/Model changes (SVM vs VQC vs QSVM)

Usage:
    python create_focused_comparisons.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data_loader import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_all_results():
    """Load all result files."""
    results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results')
    
    results = {}
    
    # Baseline results
    baseline_path = os.path.join(results_dir, 'svm_baseline_results.csv')
    if os.path.exists(baseline_path):
        results['baseline'] = pd.read_csv(baseline_path)
    
    # Parameter comparison
    param_path = os.path.join(results_dir, 'svm_parameter_comparison_results.csv')
    if os.path.exists(param_path):
        results['parameter'] = pd.read_csv(param_path)
    
    # Hyperparameter tuning
    hyper_path = os.path.join(results_dir, 'svm_hyperparameter_tuning_results.csv')
    if os.path.exists(hyper_path):
        results['hyperparameter'] = pd.read_csv(hyper_path)
    
    # VQC results
    vqc_path = os.path.join(results_dir, 'vqc_hyperparameter_tuning_results.csv')
    if os.path.exists(vqc_path):
        results['vqc'] = pd.read_csv(vqc_path)
    
    return results


def create_parameter_comparison(results):
    """Create simple parameter comparison visualization."""
    if 'parameter' not in results:
        print("No parameter comparison results found.")
        return
    
    df = results['parameter']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SVM Parameter Effects Comparison', fontsize=16, fontweight='bold')
    
    # 1. C Parameter Effect
    ax1 = axes[0, 0]
    c_effect = df.groupby('C')['accuracy'].mean().sort_index()
    bars = ax1.bar(range(len(c_effect)), c_effect.values, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(c_effect)))
    ax1.set_xticklabels([f"C={c}" for c in c_effect.index], fontsize=11)
    ax1.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of C Parameter', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0.7, 0.8])
    for i, (bar, val) in enumerate(zip(bars, c_effect.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Gamma Parameter Effect
    ax2 = axes[0, 1]
    gamma_effect = df.groupby('gamma')['accuracy'].mean().sort_values(ascending=False)
    bars = ax2.bar(range(len(gamma_effect)), gamma_effect.values, color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(gamma_effect)))
    ax2.set_xticklabels([str(g)[:8] if len(str(g)) > 8 else str(g) for g in gamma_effect.index], 
                        rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Effect of Gamma Parameter', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0.7, 0.8])
    for i, (bar, val) in enumerate(zip(bars, gamma_effect.values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Kernel Type Effect
    ax3 = axes[1, 0]
    kernel_effect = df.groupby('kernel')['accuracy'].mean().sort_values(ascending=False)
    bars = ax3.bar(range(len(kernel_effect)), kernel_effect.values, color='lightgreen', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(kernel_effect)))
    ax3.set_xticklabels([k.upper() for k in kernel_effect.index], fontsize=11)
    ax3.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Effect of Kernel Type', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim([0.6, 0.8])
    for i, (bar, val) in enumerate(zip(bars, kernel_effect.values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Best Configuration per Dataset
    ax4 = axes[1, 1]
    best_configs = df.loc[df.groupby('dataset')['accuracy'].idxmax()]
    best_configs = best_configs.sort_values('accuracy', ascending=True)
    y_pos = np.arange(len(best_configs))
    bars = ax4.barh(y_pos, best_configs['accuracy'], color='mediumpurple', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([d.replace('_', ' ').title() for d in best_configs['dataset']], fontsize=11)
    ax4.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Best Configuration per Dataset', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    for i, (bar, row) in enumerate(zip(bars, best_configs.itertuples())):
        config_text = f"C={row.C}, γ={str(row.gamma)[:6]}, k={row.kernel[:4]}"
        ax4.text(0.01, bar.get_y() + bar.get_height()/2, config_text,
                va='center', fontsize=9, color='white', fontweight='bold')
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{row.accuracy:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results/figures')
    save_path = os.path.join(results_dir, 'parameter_comparison_simple.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Parameter comparison saved to: {save_path}")


def create_qubit_comparison(results):
    """Create qubit comparison visualization."""
    if 'baseline' not in results:
        print("No baseline results found.")
        return
    
    df = results['baseline']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Qubit Configuration Effects Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy by Qubit Count
    ax1 = axes[0, 0]
    qubit_effect = df.groupby('n_qubits')['accuracy'].mean().sort_index()
    bars = ax1.bar(range(len(qubit_effect)), qubit_effect.values, color='teal', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(qubit_effect)))
    ax1.set_xticklabels([f"{q} Qubits" for q in qubit_effect.index], fontsize=11)
    ax1.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Average Accuracy by Qubit Count', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars, qubit_effect.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Accuracy by Dataset and Qubits
    ax2 = axes[0, 1]
    pivot = df.pivot_table(values='accuracy', index='dataset', columns='n_qubits', aggfunc='mean')
    pivot = pivot.sort_values(pivot.columns[0], ascending=True)
    pivot.plot(kind='barh', ax=ax2, width=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
              edgecolor='black', linewidth=1)
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Dataset and Qubit Configuration', fontsize=13, fontweight='bold')
    ax2.legend(title='Qubits', fontsize=10, title_fontsize=11)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_yticklabels([d.replace('_', ' ').title() for d in pivot.index], fontsize=10)
    
    # 3. Training Time by Qubit Count
    ax3 = axes[1, 0]
    time_effect = df.groupby('n_qubits')['training_time'].mean().sort_index()
    bars = ax3.bar(range(len(time_effect)), time_effect.values, color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(time_effect)))
    ax3.set_xticklabels([f"{q} Qubits" for q in time_effect.index], fontsize=11)
    ax3.set_ylabel('Average Training Time (s)', fontsize=12, fontweight='bold')
    ax3.set_title('Training Time by Qubit Count', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars, time_effect.values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(time_effect.values)*0.02,
                f'{val:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. F1-Score by Qubit Count
    ax4 = axes[1, 1]
    f1_effect = df.groupby('n_qubits')['f1_score'].mean().sort_index()
    bars = ax4.bar(range(len(f1_effect)), f1_effect.values, color='salmon', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(range(len(f1_effect)))
    ax4.set_xticklabels([f"{q} Qubits" for q in f1_effect.index], fontsize=11)
    ax4.set_ylabel('Average F1-Score', fontsize=12, fontweight='bold')
    ax4.set_title('F1-Score by Qubit Count', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars, f1_effect.values)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results/figures')
    save_path = os.path.join(results_dir, 'qubit_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Qubit comparison saved to: {save_path}")


def create_model_comparison(results):
    """Create model comparison (SVM vs VQC)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison: SVM vs VQC', fontsize=16, fontweight='bold')
    
    # Prepare data
    if 'baseline' in results and 'vqc' in results:
        svm_df = results['baseline']
        vqc_df = results['vqc']
        
        # Get common datasets
        svm_datasets = set(svm_df['dataset'].unique())
        vqc_datasets = set(vqc_df['dataset'].unique())
        common_datasets = list(svm_datasets & vqc_datasets)
        
        if len(common_datasets) > 0:
            # 1. Accuracy Comparison
            ax1 = axes[0]
            svm_acc = [svm_df[svm_df['dataset'] == ds]['accuracy'].max() for ds in common_datasets]
            vqc_acc = [vqc_df[vqc_df['dataset'] == ds]['accuracy'].max() for ds in common_datasets]
            
            x = np.arange(len(common_datasets))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, svm_acc, width, label='SVM', color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
            bars2 = ax1.bar(x + width/2, vqc_acc, width, label='VQC', color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
            
            ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
            ax1.set_title('Best Accuracy Comparison', fontsize=13, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([d.replace('_', ' ').title() for d in common_datasets], fontsize=10)
            ax1.legend(fontsize=11)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 2. Training Time Comparison
            ax2 = axes[1]
            svm_time = [svm_df[svm_df['dataset'] == ds]['training_time'].mean() for ds in common_datasets]
            vqc_time = [vqc_df[vqc_df['dataset'] == ds]['training_time'].mean() for ds in common_datasets]
            
            bars1 = ax2.bar(x - width/2, svm_time, width, label='SVM', color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
            bars2 = ax2.bar(x + width/2, vqc_time, width, label='VQC', color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
            
            ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Average Training Time (s)', fontsize=12, fontweight='bold')
            ax2.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([d.replace('_', ' ').title() for d in common_datasets], fontsize=10)
            ax2.legend(fontsize=11)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.set_yscale('log')  # Log scale for better visualization
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                            f'{height:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results/figures')
    save_path = os.path.join(results_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison saved to: {save_path}")


def create_top_confusion_matrices(results):
    """Create top 6 confusion matrices in a grid."""
    loader = DataLoader()
    
    if 'parameter' not in results:
        print("No parameter comparison results found.")
        return
    
    df = results['parameter']
    
    # Get top configurations per dataset
    datasets = ['breast_cancer', 'iris', 'mnist']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Top Confusion Matrices - Best Configuration per Dataset', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, dataset_name in enumerate(datasets):
        dataset_df = df[df['dataset'] == dataset_name]
        
        if len(dataset_df) == 0:
            continue
        
        # Get best configuration
        best = dataset_df.loc[dataset_df['accuracy'].idxmax()]
        
        try:
            # Load data
            X_train, X_test, y_train, y_test = loader.load_dataset(dataset_name, n_qubits=int(best['n_qubits']))
            
            # Create SVM
            gamma_val = best['gamma']
            if isinstance(gamma_val, str) and gamma_val not in ['scale', 'auto']:
                try:
                    gamma_val = float(gamma_val)
                except:
                    gamma_val = 'scale'
            
            svm = SVC(C=float(best['C']), gamma=gamma_val, kernel=str(best['kernel']))
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                       xticklabels=[f'Class {i}' for i in range(len(cm))],
                       yticklabels=[f'Class {i}' for i in range(len(cm))])
            
            title = f"{dataset_name.replace('_', ' ').title()}\n"
            title += f"C={best['C']}, γ={str(best['gamma'])[:6]}, k={best['kernel'][:4]}\n"
            title += f"Accuracy: {best['accuracy']*100:.1f}%"
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Hide unused subplots
    for idx in range(len(datasets), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../../5_Results/figures')
    save_path = os.path.join(results_dir, 'top_confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Top confusion matrices saved to: {save_path}")


def main():
    """Main function."""
    print("=" * 80)
    print("Creating Focused Comparison Visualizations")
    print("=" * 80)
    
    # Load results
    results = load_all_results()
    
    if not results:
        print("ERROR: No results found. Please run experiments first.")
        return
    
    print(f"\nLoaded results: {list(results.keys())}")
    
    # Create visualizations
    print("\n1. Creating parameter comparison...")
    create_parameter_comparison(results)
    
    print("\n2. Creating qubit comparison...")
    create_qubit_comparison(results)
    
    print("\n3. Creating model comparison...")
    create_model_comparison(results)
    
    print("\n4. Creating top confusion matrices...")
    create_top_confusion_matrices(results)
    
    print("\n" + "=" * 80)
    print("All focused visualizations created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

