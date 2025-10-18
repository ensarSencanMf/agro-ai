"""
AgroAI - Model Mimarisi
Transfer Learning ile bitki hastalık tespiti için CNN modeli
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, 
    BatchNormalization, Activation
)
from tensorflow.keras.applications import (
    ResNet50, EfficientNetB0, MobileNetV2
)
from tensorflow.keras.optimizers import Adam
from src.config import ModelConfig

class PlantDiseaseModel:
    """Bitki hastalık tespiti için CNN modeli"""
    
    def __init__(self, num_classes, base_model_name='ResNet50'):
        """
        Args:
            num_classes (int): Sınıf sayısı (hastalık türü sayısı)
            base_model_name (str): Temel model adı (ResNet50, EfficientNetB0, MobileNetV2)
        """
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.config = ModelConfig()
        self.model = None
        
    def build_model(self, trainable_base=False):
        """
        Transfer learning ile model oluşturur
        
        Args:
            trainable_base (bool): Temel modelin eğitilebilir olup olmayacağı
            
        Returns:
            tensorflow.keras.Model: Oluşturulan model
        """
        # Temel model seçimi
        base_models = {
            'ResNet50': ResNet50,
            'EfficientNetB0': EfficientNetB0,
            'MobileNetV2': MobileNetV2
        }
        
        if self.base_model_name not in base_models:
            raise ValueError(f"Desteklenmeyen model: {self.base_model_name}")
        
        # Önceden eğitilmiş modeli yükle
        base_model = base_models[self.base_model_name](
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS)
        )
        
        # Temel modeli dondur (transfer learning)
        base_model.trainable = trainable_base
        
        # Yeni model katmanları ekle
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            
            # İlk Dense katmanı
            Dense(self.config.DENSE_UNITS),
            BatchNormalization(),
            Activation('relu'),
            Dropout(self.config.DROPOUT_RATE),
            
            # İkinci Dense katmanı
            Dense(self.config.DENSE_UNITS // 2),
            BatchNormalization(),
            Activation('relu'),
            Dropout(self.config.DROPOUT_RATE / 2),
            
            # Çıkış katmanı
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        
        print(f"✅ Model oluşturuldu: {self.base_model_name}")
        print(f"   Toplam parametreler: {model.count_params():,}")
        print(f"   Eğitilebilir parametreler: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def compile_model(self, learning_rate=None):
        """
        Modeli derler
        
        Args:
            learning_rate (float): Öğrenme oranı
        """
        if self.model is None:
            raise ValueError("Önce build_model() çağrılmalı!")
        
        lr = learning_rate or self.config.LEARNING_RATE
        
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"✅ Model derlendi. Learning rate: {lr}")
        
    def get_model_summary(self):
        """Model özetini yazdırır"""
        if self.model is None:
            raise ValueError("Önce build_model() çağrılmalı!")
        
        return self.model.summary()
    
    def save_model(self, filepath):
        """
        Modeli kaydeder
        
        Args:
            filepath (str): Kaydedilecek dosya yolu
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok!")
        
        self.model.save(filepath)
        print(f"✅ Model kaydedildi: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Kaydedilmiş modeli yükler
        
        Args:
            filepath (str): Model dosya yolu
            
        Returns:
            tensorflow.keras.Model: Yüklenen model
        """
        model = tf.keras.models.load_model(filepath)
        print(f"✅ Model yüklendi: {filepath}")
        return model
    
    def fine_tune_model(self, num_layers_to_unfreeze=20):
        """
        Fine-tuning için bazı katmanları eğitilebilir hale getirir
        
        Args:
            num_layers_to_unfreeze (int): Eğitilebilir hale getirilecek katman sayısı
        """
        if self.model is None:
            raise ValueError("Önce build_model() çağrılmalı!")
        
        # Base model'i al
        base_model = self.model.layers[0]
        
        # Tüm katmanları dondur
        base_model.trainable = True
        
        # Sadece son N katmanı eğitilebilir yap
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        
        print(f"✅ Fine-tuning için {num_layers_to_unfreeze} katman eğitilebilir hale getirildi")
        print(f"   Eğitilebilir parametreler: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")

# Test kodu
if __name__ == "__main__":
    # Örnek kullanım
    model_builder = PlantDiseaseModel(num_classes=38, base_model_name='ResNet50')
    model = model_builder.build_model(trainable_base=False)
    model_builder.compile_model()
    model_builder.get_model_summary()