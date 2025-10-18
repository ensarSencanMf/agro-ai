"""
AgroAI - Tahmin Modülü
Eğitilmiş model ile hastalık tespiti yapar
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

from src.config import ModelConfig, DISEASE_NAMES, TREATMENT_RECOMMENDATIONS

class PlantDiseasePredictor:
    """Bitki hastalık tahmin sınıfı"""
    
    def __init__(self, model_path):
        self.config = ModelConfig()
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model yüklendi: {model_path}")
        
    def load_and_preprocess_image(self, image_path):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(self.config.IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path, top_k=3):
        img_array = self.load_and_preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = list(DISEASE_NAMES.keys())[idx] if len(DISEASE_NAMES) > idx else f"Class_{idx}"
            confidence = float(predictions[idx])
            
            disease_info = {
                'class_name': class_name,
                'disease_name_tr': DISEASE_NAMES.get(class_name, class_name),
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.2f}%"
            }
            
            if class_name in TREATMENT_RECOMMENDATIONS:
                disease_info['treatment_tr'] = TREATMENT_RECOMMENDATIONS[class_name]['tr']
            
            results.append(disease_info)
        
        return {
            'top_prediction': results[0],
            'all_predictions': results,
            'image_path': str(image_path)
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Kullanım: python predict.py <model_path> <image_path>")
        sys.exit(1)
    
    predictor = PlantDiseasePredictor(sys.argv[1])
    result = predictor.predict(sys.argv[2])
    
    print("\n🔍 Tahmin Sonucu:")
    print(f"   Hastalık: {result['top_prediction']['disease_name_tr']}")
    print(f"   Güven: {result['top_prediction']['confidence_percent']}")
