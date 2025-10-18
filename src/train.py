"""
AgroAI - Model Eğitim Scripti
Bitki hastalık tespiti modelini eğitir
"""
import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from datetime import datetime
from pathlib import Path

from src.model import PlantDiseaseModel
from src.data_loader import PlantVillageDataLoader
from src.config import ModelConfig, SAVED_MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR

class ModelTrainer:
    """Model eğitim sınıfı"""
    
    def __init__(self, base_model_name='ResNet50'):
        self.config = ModelConfig()
        self.base_model_name = base_model_name
        self.model_builder = None
        self.model = None
        self.history = None
        
        # Zaman damgası (unique model isimleri için)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def prepare_data(self):
        """
        Veri setini hazırlar
        
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        print("⏳ Veri seti hazırlanıyor...")
        data_loader = PlantVillageDataLoader()
        train_gen, val_gen, test_gen = data_loader.create_data_generators()
        
        self.num_classes = train_gen.num_classes
        self.class_names = list(train_gen.class_indices.keys())
        
        print(f"✅ Veri seti hazır!")
        print(f"   Sınıf sayısı: {self.num_classes}")
        
        return train_gen, val_gen, test_gen
    
    def build_and_compile_model(self, trainable_base=False):
        """
        Model oluşturur ve derler
        
        Args:
            trainable_base (bool): Base model eğitilebilir mi?
        """
        print(f"\n🛠️ Model oluşturuluyor: {self.base_model_name}")
        
        self.model_builder = PlantDiseaseModel(
            num_classes=self.num_classes,
            base_model_name=self.base_model_name
        )
        
        self.model = self.model_builder.build_model(trainable_base=trainable_base)
        self.model_builder.compile_model()
        
        print("\n📋 Model Özeti:")
        self.model_builder.get_model_summary()
        
    def get_callbacks(self, stage='initial'):
        """
        Eğitim callback'lerini oluşturur
        
        Args:
            stage (str): Eğitim aşaması ('initial' veya 'finetune')
            
        Returns:
            list: Callback listesi
        """
        # Dosya yolları
        checkpoint_path = CHECKPOINTS_DIR / f"{self.base_model_name}_{stage}_{self.timestamp}.h5"
        log_dir = LOGS_DIR / f"{self.base_model_name}_{stage}_{self.timestamp}"
        csv_log_path = LOGS_DIR / f"training_{stage}_{self.timestamp}.csv"
        
        callbacks = [
            # Model checkpoint - en iyi modeli kaydet
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping - overfitting'i önle
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV logger
            CSVLogger(
                filename=str(csv_log_path),
                separator=',',
                append=False
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=None, stage='initial'):
        """
        Modeli eğitir
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Epoch sayısı
            stage (str): Eğitim aşaması
            
        Returns:
            History: Eğitim geçmişi
        """
        epochs = epochs or self.config.EPOCHS
        
        print(f"\n🚀 Eğitim başlıyor ({stage} stage)...")
        print(f"   Epoch sayısı: {epochs}")
        print(f"   Batch size: {self.config.BATCH_SIZE}")
        
        callbacks = self.get_callbacks(stage=stage)
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Eğitim tamamlandı!")
        
        return self.history
    
    def fine_tune(self, train_generator, val_generator, epochs=20, num_layers=20):
        """
        Fine-tuning yapar
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Fine-tuning epoch sayısı
            num_layers (int): Eğitilebilir hale getirilecek katman sayısı
        """
        print(f"\n🔧 Fine-tuning başlıyor...")
        
        # Bazı katmanları unfreeze et
        self.model_builder.fine_tune_model(num_layers_to_unfreeze=num_layers)
        
        # Düşük learning rate ile yeniden derle
        self.model_builder.compile_model(learning_rate=self.config.LEARNING_RATE / 10)
        
        # Fine-tuning eğitimi
        self.history = self.train(
            train_generator,
            val_generator,
            epochs=epochs,
            stage='finetune'
        )
        
        return self.history
    
    def evaluate(self, test_generator):
        """
        Modeli test eder
        
        Args:
            test_generator: Test data generator
            
        Returns:
            dict: Evaluation sonuçları
        """
        print("\n📊 Model değerlendiriliyor...")
        
        results = self.model.evaluate(test_generator, verbose=1)
        
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        print("\n✅ Test Sonuçları:")
        for name, value in metrics.items():
            print(f"   {name}: {value:.4f}")
        
        return metrics
    
    def save_final_model(self, filename=None):
        """
        Final modeli kaydeder
        
        Args:
            filename (str): Dosya adı (opsiyonel)
        """
        if filename is None:
            filename = f"agro_ai_{self.base_model_name}_{self.timestamp}.h5"
        
        filepath = SAVED_MODELS_DIR / filename
        self.model_builder.save_model(str(filepath))
        
        print(f"\n🎉 Model kaydedildi: {filepath}")
        
        return filepath

def main():
    """
    Ana eğitim fonksiyonu
    """
    print("🌾 AgroAI - Model Eğitimi Başlatılıyor...\n")
    
    # Trainer oluştur
    trainer = ModelTrainer(base_model_name='ResNet50')
    
    # Veriyi hazırla
    train_gen, val_gen, test_gen = trainer.prepare_data()
    
    # Model oluştur
    trainer.build_and_compile_model(trainable_base=False)
    
    # İlk eğitim (frozen base model)
    print("\n" + "="*60)
    print("PHASE 1: Initial Training (Frozen Base Model)")
    print("="*60)
    trainer.train(train_gen, val_gen, epochs=30, stage='initial')
    
    # Fine-tuning
    print("\n" + "="*60)
    print("PHASE 2: Fine-Tuning (Unfrozen Layers)")
    print("="*60)
    trainer.fine_tune(train_gen, val_gen, epochs=20, num_layers=20)
    
    # Test
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    test_results = trainer.evaluate(test_gen)
    
    # Modeli kaydet
    trainer.save_final_model()
    
    print("\n🎉 Tüm işlem tamamlandı!")
    print(f"   Final Test Accuracy: {test_results['accuracy']:.2%}")
    print(f"   Final Test Precision: {test_results['precision']:.2%}")
    print(f"   Final Test Recall: {test_results['recall']:.2%}")

if __name__ == "__main__":
    # GPU kontrolü
    print("GPU Durumu:")
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   GPU mevcut: {tf.config.list_physical_devices('GPU')}")
    print()
    
    main()