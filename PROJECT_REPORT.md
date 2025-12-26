# Quantum Machine Learning Algorithms - Comparative Performance Evaluation

**Bitirme Projesi Raporu**  
**Proje**: Kuantum Makine Öğrenmesi Algoritmalarının Qiskit Platformu Üzerinde Karşılaştırmalı Performans Değerlendirmesi

---

## 📋 Proje Özeti

Bu proje, klasik ve kuantum makine öğrenmesi algoritmalarının performanslarını karşılaştırmak amacıyla geliştirilmiştir. Qiskit platformu kullanılarak, 6 farklı veri seti üzerinde klasik SVM (Support Vector Machine), VQC (Variational Quantum Classifier) ve QSVM (Quantum Support Vector Machine) algoritmaları test edilmiştir.

---

## 🎯 Proje Hedefleri

1. **Klasik ve Kuantum ML Algoritmalarını Karşılaştırma**: Klasik SVM ile kuantum tabanlı VQC ve QSVM algoritmalarının performanslarını karşılaştırmak
2. **Farklı Veri Setlerinde Test**: 6 farklı veri seti üzerinde algoritmaların genel performansını değerlendirmek
3. **Qubit Sayısının Etkisini İnceleme**: 2, 4 ve 8 qubit konfigürasyonlarında algoritma performanslarını analiz etmek
4. **Reproducible Research**: Tüm deneylerin tekrarlanabilir olmasını sağlamak

---

## 📊 Kullanılan Veri Setleri

Projede 6 farklı veri seti kullanılmıştır:

| Veri Seti | Açıklama | Sınıf Sayısı | Özellik Sayısı | Qubit Konfigürasyonları |
|-----------|----------|--------------|----------------|------------------------|
| **MNIST** | El yazısı rakam tanıma | 10 | 784 (28×28) | 2, 4, 8 |
| **USGS Earthquake** | Deprem verileri | 2 | 5 | 2, 4 |
| **Breast Cancer** | Meme kanseri teşhisi | 2 | 30 | 2, 4, 8 |
| **Iris** | Çiçek sınıflandırma | 3 | 4 | 2, 4 |
| **UCI Recgym** | Sensör/IMU verileri | 3 | 561 | 2, 4, 8 |
| **PennyLane** | Kuantum ilhamlı sentetik veri | 2 | 6 | 2, 4 |

### Veri Ön İşleme

Tüm veri setleri aşağıdaki adımlardan geçirilmiştir:

1. **Eksik Değer İşleme**: Eksik değerler sütun ortalamaları ile doldurulmuştur
2. **Özellik Ölçeklendirme**: StandardScaler kullanılarak özellikler normalize edilmiştir
3. **Boyut Azaltma (PCA)**: Kuantum devrelerle uyumluluk için PCA ile 2, 4 veya 8 boyuta indirgenmiştir
4. **Veri Bölme**: %70 eğitim, %30 test olarak bölünmüştür (random_state=42)

---

## 🔬 Uygulanan Algoritmalar

### 1. Klasik SVM (Support Vector Machine)

**Amaç**: Kuantum algoritmalar için baseline performans ölçümü

**Konfigürasyon**:
- Kernel: RBF (Radial Basis Function)
- C: 1.0
- Gamma: 'scale'

**Sonuçlar**: `5_Results/svm_baseline_results.csv`

### 2. VQC (Variational Quantum Classifier)

**Amaç**: Kuantum sinir ağı yaklaşımı ile sınıflandırma

**Konfigürasyon**:
- Feature Map: ZZFeatureMap (reps=2)
- Ansatz: RealAmplitudes (reps=3)
- Optimizer: COBYLA (maxiter=100)

**Sonuçlar**: `5_Results/vqc_results.csv`

### 3. QSVM (Quantum Support Vector Machine)

**Amaç**: Kuantum kernel matrisleri kullanarak SVM sınıflandırması

**Konfigürasyon**:
- Feature Map: ZZFeatureMap (reps=2)
- Kernel: FidelityQuantumKernel
- SVM: Precomputed kernel (C=1.0, gamma='scale')

**Sonuçlar**: `5_Results/qsvm_results.csv`

---

## 📈 Deney Sonuçları

### Klasik SVM Baseline Sonuçları

| Veri Seti | Qubit | Accuracy | Precision | Recall | F1-Score | Eğitim Süresi |
|-----------|-------|----------|-----------|--------|----------|---------------|
| **Breast Cancer** | 2 | 95.91% | 95.93% | 95.91% | 95.89% | 0.004s |
| **Breast Cancer** | 4 | 95.32% | 95.48% | 95.32% | 95.27% | 0.002s |
| **Breast Cancer** | 8 | **97.08%** | **97.07%** | **97.08%** | **97.07%** | 0.002s |
| **Earthquake** | 2 | 96.35% | 96.15% | 96.35% | 96.15% | 0.147s |
| **Earthquake** | 4 | **98.12%** | **98.08%** | **98.12%** | **98.09%** | 0.096s |
| **Iris** | 2 | 93.33% | 93.45% | 93.33% | 93.33% | 0.002s |
| **Iris** | 4 | 93.33% | 93.45% | 93.33% | 93.33% | 0.002s |
| **MNIST** | 2 | 46.83% | 47.31% | 46.83% | 44.57% | 0.051s |
| **MNIST** | 4 | 63.50% | 63.81% | 63.50% | 62.97% | 0.038s |
| **MNIST** | 8 | **85.50%** | **85.67%** | **85.50%** | **85.48%** | 0.034s |
| **PennyLane** | 2 | 56.11% | 57.40% | 56.11% | 50.75% | 0.018s |
| **PennyLane** | 4 | 58.89% | 59.11% | 58.89% | 57.38% | 0.018s |
| **Recgym** | 2 | 37.56% | 36.59% | 37.56% | 31.81% | 0.045s |
| **Recgym** | 4 | 33.56% | 32.96% | 33.56% | 31.70% | 0.040s |
| **Recgym** | 8 | 35.33% | 34.58% | 35.33% | 34.25% | 0.046s |

### Önemli Bulgular

1. **En İyi Performans**: 
   - Earthquake (4 qubits): %98.12 accuracy
   - Breast Cancer (8 qubits): %97.08 accuracy
   - MNIST (8 qubits): %85.50 accuracy

2. **Qubit Sayısının Etkisi**:
   - **MNIST**: Qubit sayısı arttıkça performans önemli ölçüde artmıştır (2→4→8: 46%→63%→85%)
   - **Breast Cancer**: 8 qubit konfigürasyonu en iyi sonucu vermiştir
   - **Earthquake**: 4 qubit yeterli olmuştur

3. **Zor Veri Setleri**:
   - **Recgym**: Düşük performans (35% civarı) - karmaşık sensör verileri
   - **PennyLane**: Orta performans (56-59%) - sentetik veri seti

---

## 🏗️ Proje Yapısı

```
QML/
├── 1_Data/                          # Veri pipeline ve veri setleri
│   ├── data_preparation.py          # Veri hazırlama script'i
│   ├── raw/                         # Ham veri setleri (CSV)
│   ├── processed/                   # İşlenmiş veri setleri (NumPy)
│   └── README.md                    # Veri pipeline dokümantasyonu
│
├── 2_Notebooks/                     # Deney script'leri
│   ├── 01_Classical_Baselines/     # Klasik ML baseline deneyleri
│   │   └── svm_baseline.py        # Klasik SVM implementasyonu
│   └── 02_QML_Experiments/         # Kuantum ML deneyleri
│       ├── vqc_experiment.py       # VQC implementasyonu
│       └── qsvm_experiment.py      # QSVM implementasyonu
│
├── 3_Research/                      # Araştırma dokümantasyonu
│   ├── SVM_and_QSVM/              # SVM ve QSVM araştırma notları
│   ├── VQC_and_NeuralNetworks/     # VQC araştırma notları
│   └── Literature/                 # Literatür taraması
│
├── 4_Reports/                       # Tez raporları
│   ├── vize_raporu/                # Vize raporu
│   └── final_thesis/               # Final tez dokümanı
│
├── 5_Results/                       # Deney sonuçları
│   ├── figures/                     # Görselleştirmeler
│   ├── tables/                      # Karşılaştırma tabloları
│   ├── metrics/                     # Detaylı metrikler
│   └── *.csv                        # Sonuç CSV dosyaları
│
├── src/                             # Kaynak kod modülleri
│   ├── __init__.py
│   └── data_loader.py               # Birleşik veri yükleme arayüzü
│
├── requirements.txt                 # Python bağımlılıkları
├── README.md                        # Ana proje dokümantasyonu
└── PROJECT_REPORT.md                # Bu rapor
```

---

## 🛠️ Teknolojiler ve Kütüphaneler

### Ana Kütüphaneler

- **Qiskit** (v1.4.5): Kuantum hesaplama framework'ü
- **Qiskit Machine Learning** (v0.8.4): Kuantum ML algoritmaları
- **Qiskit Algorithms** (v0.4.0): Kuantum algoritmaları ve optimizasyon
- **Qiskit Aer** (v0.17.2): Kuantum simülatörü
- **scikit-learn** (v1.8.0): Klasik ML algoritmaları
- **NumPy** (v2.3.5): Sayısal hesaplamalar
- **Pandas** (v2.3.3): Veri manipülasyonu
- **Matplotlib**: Görselleştirme

Tüm bağımlılıklar `requirements.txt` dosyasında listelenmiştir.

---

## 🔄 Çalıştırma Adımları

### 1. Ortam Kurulumu

```bash
# Sanal ortam oluştur
python -m venv .venv

# Sanal ortamı aktifleştir
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Veri Hazırlama

```bash
python 1_Data/data_preparation.py
```

Bu script:
- 6 veri setini indirir/işler
- Eksik değerleri doldurur
- Özellikleri ölçeklendirir
- PCA ile boyut azaltır (2, 4, 8 qubits)
- Train/test split yapar
- İşlenmiş verileri `1_Data/processed/` klasörüne kaydeder

### 3. Deneyleri Çalıştırma

#### Klasik SVM Baseline

```bash
cd 2_Notebooks/01_Classical_Baselines
python svm_baseline.py
```

#### VQC Deneyleri

```bash
cd 2_Notebooks/02_QML_Experiments
python vqc_experiment.py
```

#### QSVM Deneyleri

```bash
cd 2_Notebooks/02_QML_Experiments
python qsvm_experiment.py
```

**Not**: Kuantum deneyleri uzun sürebilir (30-60 dakika). Sonuçlar otomatik olarak `5_Results/` klasörüne kaydedilir.

---

## 📊 Sonuçların Analizi

### Performans Karşılaştırması

Tüm deneyler tamamlandığında, sonuçlar `5_Results/` klasöründe bulunur:

- **CSV Dosyaları**: Her algoritma için detaylı metrikler
- **Görselleştirmeler**: Performans karşılaştırma grafikleri
- **Classification Reports**: Her konfigürasyon için detaylı sınıflandırma raporları

### Görselleştirmeler

- Accuracy karşılaştırmaları
- Training time analizleri
- F1-score karşılaştırmaları
- Quantum kernel matris görselleştirmeleri (QSVM)

---

## 🔬 Reproducibility (Tekrarlanabilirlik)

Tüm deneyler tekrarlanabilir sonuçlar üretmek için tasarlanmıştır:

- **Random Seed**: 42 (tüm rastgele işlemler için)
- **Train/Test Split**: random_state=42
- **PCA**: random_state=42
- **Sabit Konfigürasyonlar**: Tüm algoritma parametreleri sabit

Aynı ortamda aynı komutları çalıştırarak aynı sonuçları elde edebilirsiniz.

---

## 📝 Sonuçlar ve Yorumlar

### Ana Bulgular

1. **Klasik SVM Performansı**: 
   - Basit veri setlerinde (Iris, Breast Cancer) çok yüksek performans (%93-97)
   - Karmaşık veri setlerinde (MNIST) qubit sayısı arttıkça performans artıyor
   - Earthquake veri setinde en yüksek performans (%98.12)

2. **Qubit Sayısının Etkisi**:
   - Daha fazla qubit genellikle daha iyi performans sağlıyor (özellikle MNIST)
   - Ancak bazı veri setlerinde (Iris) qubit sayısının etkisi minimal

3. **Veri Seti Zorluğu**:
   - Recgym ve PennyLane veri setleri daha zorlu
   - Bu veri setlerinde tüm algoritmalar düşük performans gösteriyor

### Gelecek Çalışmalar

- Hyperparameter tuning ile performans iyileştirme
- Daha fazla qubit konfigürasyonu test etme
- Farklı feature map ve ansatz kombinasyonları deneme
- Gerçek kuantum donanımında test etme

---

## 📚 Referanslar

- Qiskit Documentation: https://qiskit.org/
- Qiskit Machine Learning: https://qiskit.org/ecosystem/machine-learning/
- Scikit-learn Documentation: https://scikit-learn.org/

---

## 👤 Proje Bilgileri

**Proje Adı**: Kuantum Makine Öğrenmesi Algoritmalarının Qiskit Platformu Üzerinde Karşılaştırmalı Performans Değerlendirmesi

**Proje Tipi**: Bitirme Tezi

**Platform**: Qiskit (Quantum Simulator)

**Tarih**: 2024

## 7. Hiperparametre Optimizasyonu ve Gelişmiş Analizler

### 7.1 Yeni Özellikler

Projeye eklenen yeni özellikler:

1. **Hiperparametre Optimizasyonu**:
   - SVM için GridSearchCV ile otomatik optimizasyon
   - VQC için farklı konfigürasyon testleri
   - Best parameter seçimi

2. **Gelişmiş Görselleştirmeler**:
   - Confusion matrices
   - ROC curves (binary classification)
   - Hyperparameter comparison heatmaps
   - Parameter vs Performance scatter plots

3. **Parametre Karşılaştırması**:
   - 12 farklı SVM konfigürasyonu
   - 5+ farklı VQC konfigürasyonu
   - Kapsamlı performans analizi

### 7.2 Yeni Script'ler

- `hyperparameter_tuning.py`: SVM için GridSearchCV optimizasyonu
- `svm_parameter_comparison.py`: Farklı parametre konfigürasyonlarını karşılaştırma
- `vqc_hyperparameter_tuning.py`: VQC için farklı konfigürasyon testleri

### 7.3 Kuantum Sınıflandırma Modelleri Raporu

Detaylı rapor: `3_Research/Quantum_Classification_Models_Report.md`

- Mevcut kuantum modeller analizi
- Klasik modellerin kuantum versiyonları
- Hybrid yaklaşımlar
- Pratik öneriler

---

**Not**: Bu proje kuantum simülatörleri üzerinde çalışmaktadır. Gerçek kuantum donanımında çalıştırmak için ek konfigürasyon gereklidir.

