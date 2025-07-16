import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import pickle
from datetime import datetime

class HandWashingDataCollector:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 데이터 수집 변수
        self.collecting = False
        self.current_gesture = None
        self.data_buffer = []
        self.frame_count = 0
        self.max_frames_per_gesture = 300  # 10초 (30fps 기준)
        
        # 데이터 저장 경로
        self.data_dir = "hand_washing_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 손 씻기 동작 정의
        self.gestures = {
            'palm_rubbing': '손바닥 비비기',
            'back_rubbing': '손등 비비기',
            'finger_gaps': '손가락 사이 비비기',
            'finger_tips': '손가락 끝 비비기',
            'thumb_rubbing': '엄지손가락 비비기',
            'wrist_rubbing': '손목 비비기'
        }
    
    def extract_hand_features(self, landmarks):
        """손 랜드마크에서 특징을 추출합니다."""
        if landmarks is None:
            return None
        
        features = []
        
        # 모든 랜드마크 좌표
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # 손가락 끝점들의 좌표
        finger_tips = [8, 12, 16, 20]  # 검지, 중지, 약지, 새끼 손가락 끝
        for tip in finger_tips:
            features.extend([landmarks[tip].x, landmarks[tip].y, landmarks[tip].z])
        
        # 손가락 중간 관절들의 좌표
        finger_mids = [6, 10, 14, 18]
        for mid in finger_mids:
            features.extend([landmarks[mid].x, landmarks[mid].y, landmarks[mid].z])
        
        # 엄지손가락 좌표
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        features.extend([thumb_tip.x, thumb_tip.y, thumb_tip.z])
        features.extend([thumb_ip.x, thumb_ip.y, thumb_ip.z])
        
        # 손목 좌표
        wrist = landmarks[0]
        features.extend([wrist.x, wrist.y, wrist.z])
        
        return features
    
    def start_collecting(self, gesture_name):
        """특정 동작의 데이터 수집을 시작합니다."""
        if gesture_name not in self.gestures:
            print(f"Unknown gesture: {gesture_name}")
            return
        
        self.collecting = True
        self.current_gesture = gesture_name
        self.data_buffer = []
        self.frame_count = 0
        
        print(f"Started collecting data for: {self.gestures[gesture_name]}")
        print("Perform the hand washing gesture for 10 seconds...")
    
    def stop_collecting(self):
        """데이터 수집을 중지합니다."""
        if self.collecting:
            self.collecting = False
            self.save_data()
            print(f"Data collection completed for: {self.gestures[self.current_gesture]}")
            print(f"Collected {len(self.data_buffer)} frames")
    
    def save_data(self):
        """수집된 데이터를 저장합니다."""
        if not self.data_buffer:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_gesture}_{timestamp}.pkl"
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            'gesture': self.current_gesture,
            'timestamp': timestamp,
            'frames': self.data_buffer,
            'frame_count': len(self.data_buffer)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Data saved to: {filepath}")
    
    def run(self):
        """데이터 수집 메인 루프"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Cannot open webcam")
            return
        
        print("Hand Washing Data Collector")
        print("Controls:")
        print("1-6: Start collecting for different gestures")
        print("Space: Stop collecting")
        print("q: Quit")
        print("\nGestures:")
        for i, (key, name) in enumerate(self.gestures.items(), 1):
            print(f"{i}: {name} ({key})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.stop_collecting()
            elif key in [ord(str(i)) for i in range(1, 7)]:
                gesture_keys = list(self.gestures.keys())
                gesture_index = key - ord('1')
                if gesture_index < len(gesture_keys):
                    self.start_collecting(gesture_keys[gesture_index])
            
            # BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # 손 랜드마크 그리기
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 데이터 수집
                    if self.collecting and self.frame_count < self.max_frames_per_gesture:
                        landmarks = hand_landmarks.landmark
                        features = self.extract_hand_features(landmarks)
                        
                        if features:
                            self.data_buffer.append(features)
                            self.frame_count += 1
                    
                    # 수집 완료 시 자동 중지
                    if self.collecting and self.frame_count >= self.max_frames_per_gesture:
                        self.stop_collecting()
            
            # UI 그리기
            self.draw_ui(frame)
            
            # 화면 표시
            cv2.imshow('Hand Washing Data Collector', frame)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_ui(self, frame):
        """UI를 그립니다."""
        height, width = frame.shape[:2]
        
        # 배경 오버레이
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 제목
        cv2.putText(frame, "Hand Washing Data Collector", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 현재 상태
        if self.collecting:
            status_text = f"Collecting: {self.gestures[self.current_gesture]}"
            cv2.putText(frame, status_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            progress = self.frame_count / self.max_frames_per_gesture
            progress_text = f"Progress: {self.frame_count}/{self.max_frames_per_gesture} ({progress:.1%})"
            cv2.putText(frame, progress_text, (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Press 1-6 to start collecting", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

if __name__ == "__main__":
    collector = HandWashingDataCollector()
    collector.run() 