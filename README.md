# 🌾 AgroAI - Akıllı Bitki Hastalık Tespit Sistemi

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**AgroAI**, yapay zeka ve derin öğrenme teknolojilerini kullanarak bitki hastalıklarını otomatik olarak tespit eden akıllı bir tarım platformudur.

## 🎯 Proje Hedefi

Çiftçilerin bitki hastalıklarını erken tespit edebilmesi ve doğru tedavi önerilerine ulaşabilmesi için yapay zeka destekli bir çözüm sunmak.

## ✨ Özellikler

- 🔍 **Hastalık Tespiti**: Bitki yapraklarından 38+ farklı hastalığı tespit eder
- 📸 **Görüntü Analizi**: Deep Learning ile yüksek doğrulukta tahmin
- 💊 **Tedavi Önerileri**: Tespit edilen hastalıklar için çözüm önerileri
- 📊 **Raporlama**: Detaylı analiz raporları
- 🌍 **Türkçe Destekli**: Türk çiftçileri için yerelleştirilmiş içerik

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)
- 4GB+ RAM (model eğitimi için)
- GPU (opsiyonel, eğitimi hızlandırır)

### Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/ensarSencanMf/agro-ai.git
cd agro-ai
```

2. **Virtual environment oluşturun:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Veri setini indirin:**
```bash
# Kaggle API kurulumu (ilk kez yapıyorsanız)
pip install kaggle

# Kaggle credentials ayarlayın (kaggle.json dosyanızı ~/.kaggle/ dizinine koyun)
# https://www.kaggle.com/settings/account adresinden API token alın

# PlantVillage veri setini indirin
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/raw/
```

## 📁 Proje Yapısı

```
agro-ai/
├── README.md                    # Proje dokümantasyonu
├── requirements.txt             # Python bağımlılıkları
├── .gitignore                   # Git ignore kuralları
├── data/
│   ├── raw/                     # Ham veri seti
│   ├── processed/               # İşlenmiş veri
│   └── README.md               # Veri seti dokümantasyonu
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Veri keşfi
│   ├── 02_model_training.ipynb        # Model eğitimi
│   └── 03_evaluation.ipynb            # Model değerlendirme
├── src/
│   ├── __init__.py
│   ├── config.py               # Konfigürasyon ayarları
│   ├── data_loader.py          # Veri yükleme fonksiyonları
│   ├── model.py                # Model mimarisi
│   ├── train.py                # Eğitim scripti
│   ├── predict.py              # Tahmin scripti
│   └── utils.py                # Yardımcı fonksiyonlar
├── models/
│   ├── saved_models/           # Eğitilmiş modeller
│   └── checkpoints/            # Model checkpoints
├── app/
│   ├── app.py                  # Streamlit web uygulaması
│   └── utils.py                # App yardımcı fonksiyonları
├── tests/
│   └── test_model.py           # Unit testler
└── docs/
    ├── project_plan.md         # Proje planı
    ├── literature_review.md    # Literatür taraması
    └── presentation.pdf        # Sunum dosyası
```

## 🔬 Kullanılan Teknolojiler

- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
- **Model Architecture**: ResNet50, EfficientNet (Transfer Learning)

## 📊 Veri Seti

**PlantVillage Dataset** kullanılmaktadır:
- 54,000+ etiketlenmiş bitki yaprağı görüntüsü
- 38 farklı hastalık sınıfı
- 14 farklı bitki türü (domates, patates, mısır, vb.)

Detaylı bilgi için: [data/README.md](data/README.md)

## 🎓 Metodoloji

1. **Veri Ön İşleme**
   - Görüntü boyutlandırma (224x224)
   - Data augmentation (döndürme, kaydırma, zoom)
   - Normalizasyon

2. **Model Mimarisi**
   - Transfer Learning (ResNet50/EfficientNet)
   - Fine-tuning
   - Dropout layers (overfitting önleme)

3. **Eğitim**
   - Loss: Categorical Crossentropy
   - Optimizer: Adam
   - Metrics: Accuracy, Precision, Recall, F1-Score

4. **Değerlendirme**
   - Train/Validation/Test split (70/15/15)
   - Confusion matrix
   - Classification report

## 📈 Hedef Performans

- ✅ Accuracy: >90%
- ✅ Precision: >88%
- ✅ Recall: >88%
- ✅ Inference Time: <2 saniye

## 🖥️ Web Uygulaması

Streamlit ile geliştirilmiş basit bir web arayüzü:

```bash
streamlit run app/app.py
```

Tarayıcınızda otomatik olarak açılacaktır (http://localhost:8501)

## 📝 Kullanım Örneği

```python
from src.predict import PlantDiseasePredictor

# Model yükle
predictor = PlantDiseasePredictor('models/saved_models/best_model.h5')

# Tahmin yap
image_path = 'test_images/tomato_leaf.jpg'
result = predictor.predict(image_path)

print(f"Hastalık: {result['disease_name']}")
print(f"Güven Skoru: {result['confidence']:.2%}")
print(f"Tedavi: {result['treatment']}")
```

## 🛣️ Roadmap

### Faz 1 (Ay 1-2) ✅ MVP
- [x] Proje kurulumu
- [ ] Veri seti indirme ve ön işleme
- [ ] İlk model eğitimi
- [ ] Basit tahmin scripti

### Faz 2 (Ay 3) 🚧 Geliştirme
- [ ] Model iyileştirme ve fine-tuning
- [ ] Web uygulaması geliştirme
- [ ] Türkçe tedavi önerileri veritabanı

### Faz 3 (Ay 4) 🎯 Tamamlama
- [ ] Dokümantasyon
- [ ] Test ve validasyon
- [ ] Sunum hazırlığı
- [ ] Demo video

### Gelecek (Post-Launch) 💡
- [ ] Mobil uygulama (React Native)
- [ ] Gerçek zamanlı tespit
- [ ] Uydu görüntüsü entegrasyonu
- [ ] Çok dilli destek

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen `CONTRIBUTING.md` dosyasını okuyun.

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👨‍💻 Geliştirici

**Ensar Sencan**
- GitHub: [@ensarSencanMf](https://github.com/ensarSencanMf)

## 🙏 Teşekkürler

- PlantVillage Dataset sağlayıcılarına
- TensorFlow ve Keras topluluğuna
- Açık kaynak topluluğuna

## 📧 İletişim

Sorularınız için GitHub Issues kullanabilirsiniz.

---

⭐ **Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**
