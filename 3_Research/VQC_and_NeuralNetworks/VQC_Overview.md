# Variational Quantum Classifier (VQC) - Genel Bakış

## Tanım

Variational Quantum Classifier (VQC), klasik sinir ağlarının kuantum versiyonudur. Veriyi kuantum devrelerine kodlar ve parametreli kuantum devreleri (ansatz) kullanarak sınıflandırma yapar.

## Temel Bileşenler

### 1. Feature Map (Özellik Haritası)
- **ZZFeatureMap**: Z dönüşleri ve entanglements kullanır
- **PauliFeatureMap**: Pauli matrisleri kullanır
- Veriyi kuantum durumuna kodlar

### 2. Ansatz (Parametreli Devre)
- **RealAmplitudes**: Gerçek parametreler, daha hızlı
- **EfficientSU2**: Daha karmaşık, daha güçlü
- Öğrenilebilir parametreler içerir

### 3. Optimizer
- **COBYLA**: Gradient-free optimizasyon
- **SPSA**: Stochastic optimizasyon
- Parametreleri optimize eder

## Avantajlar ve Dezavantajlar

### Avantajlar
- Kuantum avantajı potansiyeli
- Karmaşık veri yapılarını modelleyebilme
- Küçük veri setlerinde etkili

### Dezavantajlar
- Yavaş eğitim süresi
- Bar noise (gürültü) hassasiyeti
- Büyük veri setlerinde zorluk

