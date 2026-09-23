<div align="center">

# Quantum Machine Learning: Comparative Performance Evaluation
### Comparative Evaluation of Variational Quantum Classifiers (VQC) and Quantum Support Vector Machines (QSVM) against Classical Baselines (BSc Thesis)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.4+-6929C4.svg?style=flat-square&logo=qiskit)](https://qiskit.org/)
[![Qiskit Machine Learning](https://img.shields.io/badge/Qiskit--ML-0.8+-6929C4.svg?style=flat-square)](https://qiskit.org/ecosystem/machine-learning/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?style=flat-square&logo=scikitlearn)](https://scikit-learn.org/)
[![Reproducibility: Seed 42](https://img.shields.io/badge/reproducibility-seed%2042-success.svg?style=flat-square)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg?style=flat-square)](LICENSE)

<p align="center">
  <strong>An empirical investigation into parameterized quantum circuits (PQC) and Hilbert-space quantum kernel estimation across six diverse benchmark domains.</strong>
</p>

[Abstract](#abstract) • [Quantum Circuit Architectures](#quantum-circuit-architectures) • [Benchmark Results](#benchmark-results) • [Quickstart](#quickstart) • [Repository Structure](#repository-structure) • [Reproducibility](#reproducibility)

---

</div>

## Abstract

This repository contains the experimental framework, circuit implementations, and empirical benchmark evaluations for the undergraduate thesis: **"Comparative Performance Evaluation of Quantum Machine Learning Algorithms on the Qiskit Platform"**.

The research systematically evaluates the classification capabilities, scalability, and representational capacity of near-term Noisy Intermediate-Scale Quantum (NISQ) algorithms:
- **Variational Quantum Classifier (VQC):** A parameterized quantum neural network utilizing variational circuits optimized via classical gradient-free routines (COBYLA).
- **Quantum Support Vector Machine (QSVM):** A quantum kernel method mapping classical input vectors into an exponentially large $2^n$-dimensional Hilbert state space via non-linear quantum feature maps.
- **Classical Support Vector Machine (RBF-SVM):** Classical baseline benchmark using Radial Basis Function kernels.

Each algorithm is benchmarked across **six datasets** spanning tabular diagnostics, seismological monitoring, computer vision, kinematic sensor signals, and synthetic quantum manifolds, evaluated under **2, 4, and 8 qubit** configurations.

---

## Quantum Circuit Architectures

### 1. Quantum State Preparation (Feature Mapping)
Classical vectors $\mathbf{x} \in \mathbb{R}^d$ are encoded into quantum states $|\Phi(\mathbf{x})\rangle = U_{\Phi(\mathbf{x})}|0\rangle^{\otimes n}$ using a second-order Pauli expansion (`ZZFeatureMap`) with depth $d=2$:

$$U_{\Phi(\mathbf{x})} = \exp\left(i \sum_{j} \phi_j(\mathbf{x}) Z_j + \sum_{j < k} \phi_{jk}(\mathbf{x}) Z_j Z_k\right)$$

where data mapping functions are parameterized as $\phi_j(\mathbf{x}) = x_j$ and $\phi_{jk}(\mathbf{x}) = (\pi - x_j)(\pi - x_k)$. The non-linear pairwise entangling gates generate non-trivial quantum entanglement across adjacent qubits, rendering the resulting state space classical-simulation intractable at scale.

### 2. Variational Quantum Classifier (VQC)
The VQC model couples the `ZZFeatureMap` with a hardware-efficient parameterized variational form (`RealAmplitudes`, reps=3):

$$|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle = W(\boldsymbol{\theta}) U_{\Phi(\mathbf{x})} |0\rangle^{\otimes n}$$

- **Ansatz Layers:** Alternating single-qubit parameterized rotations $R_y(\theta)$ interspersed with linear nearest-neighbor CNOT entangling gates.
- **Measurement & Loss:** Parity measurement of observable expectation values $\langle Z_0 \rangle$ mapped to class probabilities.
- **Classical Optimization Loop:** Constrained Optimization BY Linear Approximation (COBYLA) with $100$ maximum iterations.

### 3. Quantum Support Vector Machine (QSVM)
Rather than optimizing circuit parameters iteratively, QSVM evaluates the fidelity-based quantum kernel matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$ directly on the quantum state space:

$$K_{ij} = \kappa(\mathbf{x}_i, \mathbf{x}_j) = \left| \langle \Phi(\mathbf{x}_i) | \Phi(\mathbf{x}_j) \rangle \right|^2 = \left| \langle 0^{\otimes n} | U_{\Phi(\mathbf{x}_i)}^\dagger U_{\Phi(\mathbf{x}_j)} | 0^{\otimes n} \rangle \right|^2$$

The evaluated kernel matrix is subsequently passed to a dual quadratic programming solver with regularization parameter $C=1.0$.

---

## Benchmark Results

### Baseline & Multi-Qubit Evaluation Summary

All models were evaluated on identical $70/30$ stratified train-test splits with dimensionality reduced via Principal Component Analysis (PCA) to match target register widths ($n \in \{2, 4, 8\}$).

| Dataset | Domain | Classes | Features | Qubits ($n$) | Accuracy | Precision | Recall | F1-Score | Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **USGS Earthquake** | Geophysics | 2 | 5 | 2 | 96.35% | 96.15% | 96.35% | 96.15% | 0.147s |
| **USGS Earthquake** | Geophysics | 2 | 5 | **4** | **98.12%** | **98.08%** | **98.12%** | **98.09%** | 0.096s |
| **Breast Cancer** | Medical Oncology | 2 | 30 | 2 | 95.91% | 95.93% | 95.91% | 95.89% | 0.004s |
| **Breast Cancer** | Medical Oncology | 2 | 30 | 4 | 95.32% | 95.48% | 95.32% | 95.27% | 0.002s |
| **Breast Cancer** | Medical Oncology | 2 | 30 | **8** | **97.08%** | **97.07%** | **97.08%** | **97.07%** | 0.002s |
| **Iris** | Botanical Taxonomy | 3 | 4 | 2 | 93.33% | 93.45% | 93.33% | 93.33% | 0.002s |
| **Iris** | Botanical Taxonomy | 3 | 4 | 4 | 93.33% | 93.45% | 93.33% | 93.33% | 0.002s |
| **MNIST (Sub-manifold)** | Handwritten Digits | 10 | 784 | 2 | 46.83% | 47.31% | 46.83% | 44.57% | 0.051s |
| **MNIST (Sub-manifold)** | Handwritten Digits | 10 | 784 | 4 | 63.50% | 63.81% | 63.50% | 62.97% | 0.038s |
| **MNIST (Sub-manifold)** | Handwritten Digits | 10 | 784 | **8** | **85.50%** | **85.67%** | **85.50%** | **85.48%** | 0.034s |
| **PennyLane Synthetic** | Quantum Benchmark | 2 | 6 | 2 | 56.11% | 57.40% | 56.11% | 50.75% | 0.018s |
| **PennyLane Synthetic** | Quantum Benchmark | 2 | 6 | 4 | 58.89% | 59.11% | 58.89% | 57.38% | 0.018s |
| **UCI Recgym** | IMU Kinematics | 3 | 561 | 2 | 37.56% | 36.59% | 37.56% | 31.81% | 0.045s |
| **UCI Recgym** | IMU Kinematics | 3 | 561 | 4 | 33.56% | 32.96% | 33.56% | 31.70% | 0.040s |
| **UCI Recgym** | IMU Kinematics | 3 | 561 | 8 | 35.33% | 34.58% | 35.33% | 34.25% | 0.046s |

### Key Experimental Insights
1. **Peak Classification Accuracy:** The top empirical accuracy was recorded on **USGS Earthquake (4 qubits) at 98.12%**, followed closely by **Breast Cancer (8 qubits) at 97.08%**.
2. **Dimensional Register Scaling (Qubit Depth Impact):** On complex manifolds such as MNIST, scaling qubit count from $n=2$ to $n=8$ demonstrated an accuracy progression of **$46.83\% \rightarrow 63.50\% \rightarrow 85.50\%$**, confirming the hypothesis that higher Hilbert dimensionality mitigates information bottlenecks from PCA compression.
3. **High-Noise Sensor Regimes:** Multi-axis sensor signals (UCI Recgym) exhibit high intra-class variance and non-linear noise, presenting severe challenges for standard low-depth feature maps without specialized error mitigation.

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/SuleymanToklu/QML.git
cd QML

# Initialize virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation & Normalization Pipeline

```bash
python 1_Data/data_preparation.py
```

This automated pipeline:
- Fetches and parses raw benchmark datasets.
- Imputes missing features with column-wise central tendencies.
- Standardizes feature distributions via $z$-score normalization.
- Performs deterministic PCA projections for 2, 4, and 8 qubit registers.
- Serializes prepared NumPy arrays to `1_Data/processed/`.

### 3. Executing Experiments

```bash
# Classical SVM Baseline Benchmark
python 2_Notebooks/01_Classical_Baselines/svm_baseline.py

# Variational Quantum Classifier (VQC)
python 2_Notebooks/02_QML_Experiments/vqc_experiment.py

# Quantum Support Vector Machine (QSVM)
python 2_Notebooks/02_QML_Experiments/qsvm_experiment.py
```

*Note: Execution on state-vector simulators requires approximately 30 to 60 minutes depending on qubit width and sample size. Numerical outputs and confusion matrices are automatically saved to `5_Results/`.*

---

## Programmatic Usage

```python
from src.data_loader import DataLoader

# Initialize unified dataset interface
loader = DataLoader()

# Query available benchmark registries
available_datasets = loader.list_datasets()
print("Registries:", available_datasets)

# Load dimensionally-projected register (e.g. Breast Cancer on 4 qubits)
X_train, X_test, y_train, y_test = loader.load_dataset('breast_cancer', n_qubits=4)

# Retrieve metadata and dimensionality schema
meta = loader.get_dataset_info('breast_cancer')
print(f"Loaded {meta['name']} -> Train: {X_train.shape}, Test: {X_test.shape}")
```

---

## Repository Structure

```
QML/
├── 1_Data/                          # Data pipelines and dataset artifacts
│   ├── data_preparation.py          # Automated ETL and PCA reduction script
│   ├── README.md                    # Data pipeline technical documentation
│   ├── raw/                         # Raw canonical CSV datasets
│   └── processed/                   # Dimensionally-reduced NumPy matrices
├── 2_Notebooks/                     # Experimental execution routines
│   ├── 01_Classical_Baselines/      # Classical machine learning baseline experiments
│   │   └── svm_baseline.py          # Scikit-learn RBF-SVM reference model
│   └── 02_QML_Experiments/         # Quantum algorithm implementations
│       ├── vqc_experiment.py        # Parameterized Variational Quantum Classifier
│       └── qsvm_experiment.py       # Quantum Kernel Support Vector Machine
├── 3_Research/                      # Theoretical documentation and literature notes
│   ├── SVM_and_QSVM/                # Mathematical formulation of quantum kernels
│   ├── VQC_and_NeuralNetworks/      # Variational ansatz design notes
│   ├── Literature/                  # NISQ and QML literature review
│   └── Quantum_Classification_Models_Report.md # Extended comparative research report
├── 4_Reports/                       # Academic thesis deliverables
│   ├── vize_raporu/                 # Midterm academic milestone report
│   └── final_thesis/                # Final undergraduate thesis manuscript
├── 5_Results/                       # Empirical artifacts and telemetry
│   ├── figures/                     # Performance curves, ROC plots, and heatmaps
│   ├── tables/                      # Comparative accuracy and latency matrices
│   ├── metrics/                     # Detailed classification reports
│   └── *.csv                        # Machine-readable evaluation logs
├── src/                             # Shared modular utility library
│   ├── __init__.py
│   └── data_loader.py               # Standardized dataset loading interface
├── requirements.txt                 # Pinned library dependencies
├── PROJECT_REPORT.md                # Comprehensive technical and empirical report
├── LICENSE                          # MIT License
└── README.md                        # Academic project documentation
```

---

## Reproducibility Protocol

To guarantee rigorous scientific reproducibility across heterogeneous compute platforms:
- **Global Pseudo-Random Number Seed:** Deterministically locked to `42` (`numpy.random.seed(42)`).
- **Partitioning:** Stratified $70/30$ train/test partitions generated with `random_state=42`.
- **Dimensionality Reduction:** Deterministic PCA projections via `sklearn.decomposition.PCA(random_state=42)`.
- **Quantum Simulation Backend:** Qiskit Aer `AerSimulator` configured with default deterministic state-vector propagation.

---

## Academic Context & Citation

This project was conducted as an undergraduate graduation thesis in Computer Engineering, exploring the viability of quantum kernel approximations and variational quantum optimization on near-term quantum architectures.

```bibtex
@thesis{toklu2024qml,
  author       = {Suleyman Toklu},
  title        = {Kuantum Makine Ogrenmesi Algoritmalarinin Qiskit Platformu Uzerinde Karsilastirmali Performans Degerlendirmesi},
  school       = {Department of Computer Engineering},
  year         = {2024},
  type         = {BSc Graduation Thesis},
  note         = {Comparative Evaluation of Quantum Machine Learning Algorithms using Qiskit}
}
```

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
