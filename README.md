# Quantum Machine Learning Algorithms - Comparative Performance Evaluation

**Bitirme Projesi: Kuantum Makine Öğrenmesi Algoritmalarının Qiskit Platformu Üzerinde Karşılaştırmalı Performans Değerlendirmesi**

Bu repository, Qiskit platformu kullanılarak Kuantum Makine Öğrenmesi algoritmalarının karşılaştırmalı performans değerlendirmesi için geliştirilmiş implementasyon ve deneyleri içermektedir.

## 📁 Proje Yapısı

```
QML/
├── 1_Data/                          # Veri pipeline ve veri setleri
│   ├── data_preparation.py          # Veri hazırlama script'i
│   ├── README.md                    # Veri pipeline dokümantasyonu
│   ├── raw/                         # Ham veri setleri (CSV format)
│   └── processed/                  # İşlenmiş veri setleri (NumPy format)
│
├── 2_Notebooks/                     # Deney script'leri
│   ├── 01_Classical_Baselines/      # Klasik ML baseline deneyleri
│   │   └── svm_baseline.py         # Klasik SVM implementasyonu
│   └── 02_QML_Experiments/         # Kuantum ML deneyleri
│       ├── vqc_experiment.py       # Variational Quantum Classifier
│       └── qsvm_experiment.py      # Quantum Support Vector Machine
│
├── 3_Research/                       # Araştırma dokümantasyonu
│   ├── SVM_and_QSVM/               # SVM ve QSVM araştırma notları
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
│   └── *.csv                        # Sonuç CSV dosyaları
│
├── src/                             # Kaynak kod modülleri
│   ├── __init__.py
│   └── data_loader.py              # Birleşik veri yükleme arayüzü
│
├── requirements.txt                 # Python bağımlılıkları
├── README.md                        # Bu dosya
└── PROJECT_REPORT.md                # Detaylı proje raporu
```

## 🚀 Hızlı Başlangıç

### 1. Ortam Kurulumu

```bash
# Sanal ortam oluştur (önerilir)
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

Bu script veri setlerini indirir, ön işler ve `1_Data/processed/` klasörüne kaydeder.
Detaylı bilgi için `1_Data/README.md` dosyasına bakın.

### 3. Deneyleri Çalıştırma

#### Klasik SVM Baseline
```bash
cd 2_Notebooks/01_Classical_Baselines
python svm_baseline.py
```

#### VQC (Variational Quantum Classifier) Deneyleri
```bash
cd 2_Notebooks/02_QML_Experiments
python vqc_experiment.py
```

#### QSVM (Quantum Support Vector Machine) Deneyleri
```bash
cd 2_Notebooks/02_QML_Experiments
python qsvm_experiment.py
```

**Not**: Kuantum deneyleri uzun sürebilir (30-60 dakika). Sonuçlar otomatik olarak `5_Results/` klasörüne kaydedilir.

## 📊 Veri Setleri

Projede 6 farklı veri seti kullanılmıştır:

1. **MNIST** - El yazısı rakam tanıma
2. **USGS Earthquake** - Deprem verileri
3. **UCI Recgym** - Sensör/IMU verileri
4. **PennyLane** - QML-native benchmark veri setleri
5. **Breast Cancer** - Meme kanseri teşhisi
6. **Iris** - Algoritma testi için baseline veri seti

Veri setleri ön işleme adımlarından geçirilmiş ve `1_Data/processed/` klasörüne kaydedilmiştir.
Detaylı veri hazırlama süreci için `1_Data/data_preparation.py` script'ini çalıştırın.

## 💻 Kullanım

### Veri Yükleme

```python
from src.data_loader import DataLoader

# Loader'ı başlat
loader = DataLoader()

# Mevcut veri setlerini listele
datasets = loader.list_datasets()
print(datasets)

# Bir veri setini yükle
X_train, X_test, y_train, y_test = loader.load_dataset('iris', n_qubits=2)

# Veri seti bilgilerini al
info = loader.get_dataset_info('iris')
print(info)
```

## 🔬 Deneyler

### Klasik Baseline
- **SVM (Support Vector Machine)**: Karşılaştırma için klasik baseline
- Implementasyon: `2_Notebooks/01_Classical_Baselines/svm_baseline.py`

### Kuantum ML Modelleri
- **VQC (Variational Quantum Classifier)**: Kuantum sinir ağı yaklaşımı
- **QSVM (Quantum Support Vector Machine)**: Kuantum kernel metodu
- Implementasyon: `2_Notebooks/02_QML_Experiments/`

## 📝 Tekrarlanabilirlik (Reproducibility)

Tüm deneyler tekrarlanabilir sonuçlar için sabit random seed kullanır:
- NumPy random seed: 42
- Train/test split random state: 42
- PCA random state: 42

## 📚 Dokümantasyon

- **Veri Pipeline**: `1_Data/README.md` dosyasına bakın
- **Deney Çalıştırma**: `2_Notebooks/RUN_EXPERIMENTS.md` dosyasına bakın
- **Araştırma Notları**: `3_Research/` klasörüne bakın
- **Detaylı Proje Raporu**: `PROJECT_REPORT.md` dosyasına bakın

## 🛠️ Bağımlılıklar

Ana bağımlılıklar:
- **Qiskit**: Kuantum hesaplama framework'ü
- **Qiskit Machine Learning**: Kuantum ML algoritmaları
- **Qiskit Algorithms**: Kuantum algoritmaları ve optimizasyon
- **scikit-learn**: Klasik ML algoritmaları
- **NumPy/Pandas**: Veri manipülasyonu
- **Matplotlib**: Görselleştirme

Tüm bağımlılıklar `requirements.txt` dosyasında listelenmiştir.

## 📊 Sonuçlar

Detaylı deney sonuçları ve analizler için `PROJECT_REPORT.md` dosyasına bakın.

Ana bulgular:
- **En iyi performans**: Earthquake (4 qubits) - %98.12 accuracy
- **Qubit sayısının etkisi**: MNIST'te qubit sayısı arttıkça performans önemli ölçüde artmıştır
- **Sonuç dosyaları**: `5_Results/` klasöründe CSV formatında

## 👤 Proje Bilgileri

**Proje Adı**: Kuantum Makine Öğrenmesi Algoritmalarının Qiskit Platformu Üzerinde Karşılaştırmalı Performans Değerlendirmesi

**Proje Tipi**: Bitirme Tezi

**Platform**: Qiskit (Quantum Simulator)

---

**Not**: Bu proje kuantum simülatörleri üzerinde çalışmaktadır. Gerçek kuantum donanımında çalıştırmak için ek konfigürasyon gereklidir.
