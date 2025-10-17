# 📊 Veri Seti Dokümantasyonu

## PlantVillage Dataset

### Genel Bakış
- **Toplam Görüntü**: ~54,000
- **Sınıf Sayısı**: 38 (hastalık + sağlıklı)
- **Bitki Türü**: 14 farklı bitki
- **Görüntü Boyutu**: 256x256 pixels (orijinal)
- **Format**: JPG

### Veri Yapısı

```
data/raw/PlantVillage/
├── Tomato___Bacterial_spot/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___Leaf_Mold/
├── Tomato___Septoria_leaf_spot/
├── Tomato___Spider_mites/
├── Tomato___Target_Spot/
├── Tomato___Mosaic_virus/
├── Tomato___Yellow_Leaf_Curl_Virus/
├── Tomato___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Corn___Common_rust/
├── Corn___Northern_Leaf_Blight/
├── Corn___healthy/
└── ... (38 klasör toplam)
```

### Bitki Türleri

1. 🍅 **Domates** (Tomato) - 10 sınıf
2. 🥔 **Patates** (Potato) - 3 sınıf
3. 🌽 **Mısır** (Corn) - 4 sınıf
4. 🍇 **Üzüm** (Grape) - 4 sınıf
5. 🍎 **Elma** (Apple) - 4 sınıf
6. 🫑 **Biber** (Pepper) - 2 sınıf
7. 🍑 **Şeftali** (Peach) - 2 sınıf
8. 🍒 **Kiraz** (Cherry) - 2 sınıf
9. 🍓 **Çilek** (Strawberry) - 2 sınıf
10. 🫐 **Yaban mersini** (Blueberry) - 1 sınıf
11. 🍊 **Portakal** (Orange) - 1 sınıf
12. 🫘 **Soya** (Soybean) - 1 sınıf
13. 🥒 **Kabak** (Squash) - 1 sınıf
14. 🥬 **Turp** (Raspberry) - 1 sınıf

### Veri İndirme

#### Yöntem 1: Kaggle CLI (Önerilen)

```bash
# Kaggle API kurulumu
pip install kaggle

# Kaggle API token alın:
# 1. https://www.kaggle.com/settings/account adresine gidin
# 2. "Create New API Token" butonuna tıklayın
# 3. İndirilen kaggle.json dosyasını şu dizine taşıyın:
#    - Linux/Mac: ~/.kaggle/kaggle.json
#    - Windows: C:\Users\<username>\.kaggle\kaggle.json

# Veri setini indirin
kaggle datasets download -d abdallahalidev/plantvillage-dataset

# Zip'i açın
unzip plantvillage-dataset.zip -d data/raw/
```

#### Yöntem 2: Manuel İndirme

1. https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset adresine gidin
2. "Download" butonuna tıklayın
3. İndirilen zip dosyasını `data/raw/` dizinine çıkarın

### Veri Ön İşleme

Model eğitimi öncesi uygulanacak işlemler:

1. **Boyutlandırma**: 224x224 (ResNet50 için optimal)
2. **Normalizasyon**: [0, 1] aralığına ölçekleme
3. **Data Augmentation**:
   - Rastgele döndürme (±15°)
   - Yatay çevirme
   - Zoom (0.9-1.1)
   - Parlaklık ayarı

### Train/Val/Test Split

- **Training**: 70% (~37,800 görüntü)
- **Validation**: 15% (~8,100 görüntü)
- **Test**: 15% (~8,100 görüntü)

### Sınıf Dengesizliği

Bazı sınıflarda görüntü sayısı az olabilir. Bu durumda:
- Class weights kullanımı
- Data augmentation ile dengeleme
- SMOTE gibi teknikler

### Lisans

PlantVillage dataset, akademik ve ticari kullanım için açık kaynaklıdır.

### Kaynaklar

- **Kaggle**: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **Orijinal Paper**: Hughes, D. P., & Salathe, M. (2015). "An open access repository of images on plant health to enable the development of mobile disease diagnostics.
