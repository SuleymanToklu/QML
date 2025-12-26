# 🚀 Deneyleri Çalıştırma Rehberi

Bu klasörde Python script dosyaları bulunmaktadır. Tüm deneyler script formatında çalıştırılabilir.

## 📁 Dosya Yapısı

```
2_Notebooks/
├── 01_Classical_Baselines/
│   └── svm_baseline.py          # Klasik SVM deneyleri
└── 02_QML_Experiments/
    ├── vqc_experiment.py        # VQC deneyleri
    └── qsvm_experiment.py       # QSVM deneyleri
```

## 📋 Çalıştırma Adımları

### ÖNCE: Veri Hazırlama

**Script ile:**
```bash
# Proje root klasöründen
python 1_Data/data_preparation.py
```

### 1. Klasik SVM Deneyleri

```bash
cd 2_Notebooks/01_Classical_Baselines
python svm_baseline.py
```

### 2. VQC Deneyleri

```bash
cd 2_Notebooks/02_QML_Experiments
python vqc_experiment.py
```

**Arka planda çalıştırma (Windows PowerShell):**
```bash
Start-Process python -ArgumentList "vqc_experiment.py" -WindowStyle Hidden
```

**Arka planda çalıştırma (Linux/Mac):**
```bash
nohup python vqc_experiment.py > vqc_output.log 2>&1 &
```

### 3. QSVM Deneyleri

```bash
cd 2_Notebooks/02_QML_Experiments
python qsvm_experiment.py
```

**Arka planda çalıştırma (Windows PowerShell):**
```bash
Start-Process python -ArgumentList "qsvm_experiment.py" -WindowStyle Hidden
```

**Arka planda çalıştırma (Linux/Mac):**
```bash
nohup python qsvm_experiment.py > qsvm_output.log 2>&1 &
```

## 🔄 Tüm Deneyleri Sırayla Çalıştırma

```bash
# Proje root klasöründen
cd 2_Notebooks/01_Classical_Baselines
python svm_baseline.py

cd ../02_QML_Experiments
python vqc_experiment.py
python qsvm_experiment.py
```

**Batch script (Windows):**
```batch
@echo off
cd 2_Notebooks\01_Classical_Baselines
python svm_baseline.py
cd ..\02_QML_Experiments
python vqc_experiment.py
python qsvm_experiment.py
pause
```

**Shell script (Linux/Mac):**
```bash
#!/bin/bash
cd 2_Notebooks/01_Classical_Baselines
python svm_baseline.py
cd ../02_QML_Experiments
python vqc_experiment.py
python qsvm_experiment.py
```

## ⚠️ Önemli Notlar

1. **Önce veri hazırlama**: `1_Data/data_preparation.py` mutlaka çalıştırılmalı
2. **Sıralama**: SVM → VQC → QSVM sırası önerilir (SVM en hızlı)
3. **Süre**: 
   - SVM: ~1 dakika
   - VQC: ~30-60 dakika (kuantum simülasyon yavaş)
   - QSVM: ~30-60 dakika (kuantum kernel hesaplama yavaş)
4. **Sonuçlar**: Tüm sonuçlar `5_Results/` klasörüne kaydedilir

## 🐛 Sorun Giderme

### "ModuleNotFoundError: No module named 'src'"
**Çözüm**: Proje root klasöründen çalıştırdığınızdan emin olun veya script'lerde path ayarları doğru.

### "FileNotFoundError: processed/ ... not found"
**Çözüm**: Önce `1_Data/data_preparation.py` çalıştırın.

### Script'ler görselleştirme göstermiyor
**Normal**: Script'ler `matplotlib.use('Agg')` kullanır, görseller dosyaya kaydedilir.
Görselleri görmek için `5_Results/figures/` klasöründeki PNG dosyalarını açın.

## 📊 Sonuçlar

Tüm deneyler tamamlandığında şunlar oluşur:

- `5_Results/svm_baseline_results.csv`
- `5_Results/vqc_results.csv`
- `5_Results/qsvm_results.csv`
- `5_Results/figures/svm_baseline_comparison.png`
- `5_Results/figures/vqc_comparison.png`
- `5_Results/figures/qsvm_comparison.png`
- `5_Results/figures/qsvm_kernel_matrix.png`

