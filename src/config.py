"""
AgroAI - Konfigürasyon Ayarları
"""
import os
from pathlib import Path

# Proje kök dizini
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Veri dizinleri
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model dizinleri
MODEL_DIR = ROOT_DIR / "models"
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"

# Log dizini
LOGS_DIR = ROOT_DIR / "logs"

# Dizinleri oluştur
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 MODEL_DIR, SAVED_MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model hiperparametreleri
class ModelConfig:
    # Görüntü parametreleri
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    
    # Eğitim parametreleri
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Model mimarisi
    BASE_MODEL = "ResNet50"  # ResNet50, EfficientNetB0, MobileNetV2
    DROPOUT_RATE = 0.5
    DENSE_UNITS = 256
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Seed (reproducibility için)
    RANDOM_SEED = 42

# Hastalık isimleri (Türkçe-İngilizce mapping)
DISEASE_NAMES = {
    "Tomato___Bacterial_spot": "Domates - Bakteriyel Leke Hastalığı",
    "Tomato___Early_blight": "Domates - Erken Yanıklık",
    "Tomato___Late_blight": "Domates - Geç Yanıklık",
    "Tomato___Leaf_Mold": "Domates - Yaprak Küfü",
    "Tomato___Septoria_leaf_spot": "Domates - Septoria Yaprak Lekesi",
    "Tomato___Spider_mites": "Domates - Kırmızı Örümcek",
    "Tomato___Target_Spot": "Domates - Hedef Leke Hastalığı",
    "Tomato___Mosaic_virus": "Domates - Mozaik Virüsü",
    "Tomato___Yellow_Leaf_Curl_Virus": "Domates - Sarı Yaprak Kıvırcıklığı Virüsü",
    "Tomato___healthy": "Domates - Sağlıklı",
    "Potato___Early_blight": "Patates - Erken Yanıklık",
    "Potato___Late_blight": "Patates - Geç Yanıklık",
    "Potato___healthy": "Patates - Sağlıklı",
}

# Tedavi önerileri
TREATMENT_RECOMMENDATIONS = {
    "Tomato___Bacterial_spot": {
        "tr": "Bakır bazlı fungisitler kullanın. Sulamayı sabah saatlerinde yapın.",
        "en": "Use copper-based fungicides. Water in the morning hours."
    },
    "Tomato___Early_blight": {
        "tr": "Hastalıklı yaprakları temizleyin. Fungisit uygulayın. Bitki rotasyonu yapın.",
        "en": "Remove infected leaves. Apply fungicide. Practice crop rotation."
    },
}

print(f"✅ Config yüklendi. Proje dizini: {ROOT_DIR}")