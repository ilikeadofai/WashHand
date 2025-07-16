import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from datetime import datetime

class HandWashingModelTrainer:
    def __init__(self):
        self.data_dir = "hand_washing_data"
        self.model_dir = "trained_models"
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 손 씻기 동작 정의
        self.gestures = {
            'palm_rubbing': '손바닥 비비기',
            'back_rubbing': '손등 비비기',
            'finger_gaps': '손가락 사이 비비기',
            'finger_tips': '손가락 끝 비비기',
            'thumb_rubbing': '엄지손가락 비비기',
            'wrist_rubbing': '손목 비비기'
        }
        
        self.models = {}
    
    def load_data(self):
        """수집된 데이터를 로드합니다."""
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        
        if not data_files:
            print("No data files found!")
            return None, None
        
        X = []  # 특징 데이터
        y = []  # 라벨 데이터
        
        for file in data_files:
            filepath = os.path.join(self.data_dir, file)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                gesture = data['gesture']
                frames = data['frames']
                
                # 각 프레임을 샘플로 추가
                for frame_features in frames:
                    X.append(frame_features)
                    y.append(gesture)
                
                print(f"Loaded {len(frames)} frames from {file} ({gesture})")
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        return np.array(X), np.array(y)
    
    def train_models(self):
        """여러 모델을 훈련합니다."""
        X, y = self.load_data()
        
        if X is None or len(X) == 0:
            print("No data available for training!")
            return
        
        print(f"\nTraining with {len(X)} samples")
        print(f"Features per sample: {X.shape[1]}")
        print(f"Classes: {np.unique(y)}")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 모델 정의
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        
        # 모델 훈련 및 평가
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=list(self.gestures.keys())))
            
            # 모델 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{name}_{timestamp}.joblib"
            model_path = os.path.join(self.model_dir, model_filename)
            
            joblib.dump(model, model_path)
            print(f"Model saved to: {model_path}")
            
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'path': model_path
            }
    
    def evaluate_model(self, model_name):
        """특정 모델을 평가합니다."""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        # 테스트 데이터로 재평가
        X, y = self.load_data()
        if X is None:
            return
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{model_name} Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=list(self.gestures.keys())))
    
    def predict_gesture(self, features, model_name='random_forest'):
        """새로운 특징으로 동작을 예측합니다."""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        model = self.models[model_name]['model']
        prediction = model.predict([features])[0]
        confidence = np.max(model.predict_proba([features]))
        
        return prediction, confidence

def main():
    trainer = HandWashingModelTrainer()
    
    print("Hand Washing Gesture Recognition Model Trainer")
    print("=" * 50)
    
    # 데이터 확인
    data_files = [f for f in os.listdir(trainer.data_dir) if f.endswith('.pkl')]
    print(f"Found {len(data_files)} data files")
    
    if len(data_files) == 0:
        print("\nNo data files found!")
        print("Please run data_collector.py first to collect training data.")
        return
    
    # 모델 훈련
    trainer.train_models()
    
    # 최고 성능 모델 선택
    best_model = None
    best_accuracy = 0
    
    for name, info in trainer.models.items():
        if info['accuracy'] > best_accuracy:
            best_accuracy = info['accuracy']
            best_model = name
    
    print(f"\nBest model: {best_model} (Accuracy: {best_accuracy:.4f})")
    
    # 모델 평가
    if best_model:
        trainer.evaluate_model(best_model)

if __name__ == "__main__":
    main() 