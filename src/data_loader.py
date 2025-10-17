"""
AgroAI - Veri Yükleme ve Ön İşleme
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from src.config import ModelConfig, RAW_DATA_DIR

class PlantVillageDataLoader:
    """PlantVillage veri setini yükler ve hazırlar"""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or RAW_DATA_DIR / "PlantVillage"
        self.config = ModelConfig()
        
    def create_data_generators(self):
        """
        Train, validation ve test için data generator'ları oluşturur
        
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=self.config.VAL_SPLIT + self.config.TEST_SPLIT
        )
        
        # Validation ve Test için sadece rescaling
        val_test_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.config.TEST_SPLIT / (self.config.VAL_SPLIT + self.config.TEST_SPLIT)
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=self.config.RANDOM_SEED
        )
        
        # Validation generator
        val_generator = val_test_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=False,
            seed=self.config.RANDOM_SEED
        )
        
        # Test generator
        test_generator = val_test_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=self.config.RANDOM_SEED
        )
        
        print(f"✅ Data generators oluşturuldu:")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        print(f"   Test samples: {test_generator.samples}")
        print(f"   Number of classes: {train_generator.num_classes}")
        
        return train_generator, val_generator, test_generator
    
    def get_class_names(self):
        """Sınıf isimlerini döndürür"""
        temp_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
            self.data_dir,
            target_size=self.config.IMG_SIZE,
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )
        return list(temp_gen.class_indices.keys())

# Test kodu
if __name__ == "__main__":
    loader = PlantVillageDataLoader()
    train_gen, val_gen, test_gen = loader.create_data_generators()
    print(f"\n📋 Class names: {loader.get_class_names()[:5]}...")
