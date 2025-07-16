import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from gtts import gTTS
import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
import math

class HandWashingGuide:
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
        
        # 손 씻기 단계 정의
        self.washing_steps = [
            "손바닥 비비기",
            "손등 비비기", 
            "손가락 사이 비비기",
            "손가락 끝 비비기",
            "엄지손가락 비비기",
            "손목 비비기"
        ]
        
        # 상태 변수
        self.current_step = 0
        self.step_start_time = 0
        self.total_start_time = 0
        self.is_washing = False
        self.step_completed = False
        self.washing_completed = False  # 전체 완료 상태 추가
        self.step_duration = 10  # 각 단계별 권장 시간 (초)
        self.total_duration = 60  # 전체 권장 시간 (초)
        
        # 동작 인식 변수
        self.prev_hand_landmarks = None
        self.movement_threshold = 0.05
        self.movement_counter = 0
        self.movement_threshold_count = 10
        self.current_gesture = None  # 현재 감지된 동작
        
        # 음성 파일 경로
        self.audio_files = {}
        self.create_audio_files()
        
    def create_audio_files(self):
        """음성 안내 파일들을 생성합니다."""
        print("음성 파일을 생성하는 중...")
        
        # 임시 디렉토리 생성
        if not os.path.exists("temp_audio"):
            os.makedirs("temp_audio")
        
        # 각 단계별 음성 파일 생성
        for i, step in enumerate(self.washing_steps):
            try:
                tts = gTTS(text=f"{step}를 시작합니다. 5초간 계속해주세요.", lang='ko')
                audio_path = f"temp_audio/step_{i}.mp3"
                tts.save(audio_path)
                self.audio_files[i] = audio_path
            except Exception as e:
                print(f"음성 파일 생성 실패: {e}")
        
        # 완료 메시지
        try:
            tts = gTTS(text="손 씻기가 완료되었습니다! 잘 하셨습니다.", lang='ko')
            audio_path = "temp_audio/complete.mp3"
            tts.save(audio_path)
            self.audio_files['complete'] = audio_path
        except Exception as e:
            print(f"완료 음성 파일 생성 실패: {e}")
    
    def calculate_hand_movement(self, landmarks):
        """손의 움직임을 계산합니다."""
        if self.prev_hand_landmarks is None:
            self.prev_hand_landmarks = landmarks
            return 0
        
        # 손목(0번 랜드마크)의 움직임 계산
        wrist_movement = np.linalg.norm(
            np.array([landmarks[0].x, landmarks[0].y]) - 
            np.array([self.prev_hand_landmarks[0].x, self.prev_hand_landmarks[0].y])
        )
        
        self.prev_hand_landmarks = landmarks
        return wrist_movement
    
    def detect_washing_motion(self, landmarks):
        """손 씻기 동작을 감지합니다."""
        if landmarks is None:
            return False
        
        movement = self.calculate_hand_movement(landmarks)
        
        if movement > self.movement_threshold:
            self.movement_counter += 1
        else:
            self.movement_counter = max(0, self.movement_counter - 1)
        
        # 충분한 움직임이 감지되면 손 씻기 동작으로 판단
        return self.movement_counter >= self.movement_threshold_count
    
    def detect_specific_hand_gesture(self, landmarks):
        """특정 손 씻기 동작을 감지합니다."""
        if landmarks is None:
            return None
        
        # 손가락 끝점들의 좌표 추출
        finger_tips = [8, 12, 16, 20]  # 검지, 중지, 약지, 새끼 손가락 끝
        finger_mids = [6, 10, 14, 18]  # 손가락 중간 관절
        
        # 손바닥 비비기 감지 (손가락들이 구부러져 있고 손이 좌우로 움직임)
        fingers_bent = 0
        for tip, mid in zip(finger_tips, finger_mids):
            if landmarks[tip].y > landmarks[mid].y:
                fingers_bent += 1
        
        # 손등 비비기 감지 (손가락들이 펴져 있고 손이 좌우로 움직임)
        fingers_straight = 0
        for tip, mid in zip(finger_tips, finger_mids):
            if landmarks[tip].y < landmarks[mid].y:
                fingers_straight += 1
        
        # 엄지손가락 비비기 감지 (엄지가 다른 손가락과 마찰하는 동작)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_distance = np.linalg.norm(
            np.array([thumb_tip.x, thumb_tip.y]) - 
            np.array([index_tip.x, index_tip.y])
        )
        
        # 동작 판단
        if fingers_bent >= 3 and self.movement_counter >= 5:
            return "palm_rubbing"  # 손바닥 비비기
        elif fingers_straight >= 3 and self.movement_counter >= 5:
            return "back_rubbing"  # 손등 비비기
        elif thumb_index_distance < 0.1 and self.movement_counter >= 3:
            return "thumb_rubbing"  # 엄지손가락 비비기
        else:
            return "general_movement"  # 일반적인 움직임
    
    def draw_ui(self, frame, elapsed_time, step_time):
        """UI를 그립니다."""
        height, width = frame.shape[:2]
        
        # 배경 오버레이
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 제목 (영어로 표시하여 깨짐 방지)
        cv2.putText(frame, "Hand Washing Guide System", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 현재 단계
        if self.is_washing:
            # 단계별 영어 표시
            step_names = ["Palm Rubbing", "Back Rubbing", "Finger Gaps", "Finger Tips", "Thumb Rubbing", "Wrist Rubbing"]
            current_step_text = f"Step {self.current_step + 1}: {step_names[self.current_step]}"
            cv2.putText(frame, current_step_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 진행률 표시
            progress = min(1.0, step_time / self.step_duration)
            bar_width = int(300 * progress)
            cv2.rectangle(frame, (20, 80), (320, 90), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 80), (20 + bar_width, 90), (0, 255, 0), -1)
            
            # 시간 표시
            time_text = f"Time: {step_time:.1f}s / {self.step_duration}s"
            cv2.putText(frame, time_text, (330, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 전체 시간
            total_time_text = f"Total: {elapsed_time:.1f}s / {self.total_duration}s"
            cv2.putText(frame, total_time_text, (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 전체 진행률
            total_progress = min(1.0, elapsed_time / self.total_duration)
            total_bar_width = int(400 * total_progress)
            cv2.rectangle(frame, (20, 140), (420, 150), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 140), (20 + total_bar_width, 150), (0, 255, 255), -1)
            
            # 완료 상태 표시
            if self.step_completed:
                cv2.putText(frame, "Step Complete! Press 'n' for next", (20, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Move your hands to start", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Or press 's' to start manually", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def play_audio(self, audio_path):
        """음성을 재생합니다."""
        try:
            # 간단한 방법으로 음성 파일 실행
            import platform
            if platform.system() == "Windows":
                os.system(f'start "" "{os.path.abspath(audio_path)}"')
            else:
                os.system(f'xdg-open "{audio_path}"')
        except Exception as e:
            print(f"음성 재생 실패: {e}")
            print("음성 안내 대신 텍스트로 표시됩니다.")
    
    def start_step(self, step_index):
        """새로운 단계를 시작합니다."""
        self.current_step = step_index
        self.step_start_time = time.time()
        self.step_completed = False
        self.movement_counter = 0
        
        # 음성 안내 재생
        if step_index in self.audio_files:
            threading.Thread(target=self.play_audio, args=(self.audio_files[step_index],)).start()
    
    def auto_next_step(self):
        """자동으로 다음 단계로 진행합니다."""
        if self.current_step < len(self.washing_steps) - 1:
            self.start_step(self.current_step + 1)
            print(f"Auto-progressed to step {self.current_step + 1}")
    
    def run(self):
        """메인 실행 루프"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return
        
        print("Hand Washing Guide System Started")
        print("Controls:")
        print("- Move hands to start automatically")
        print("- Press 's' to start manually")
        print("- Press 'n' to go to next step")
        print("- Press 'r' to reset")
        print("- Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and not self.is_washing and not self.washing_completed:
                # 수동 시작
                self.is_washing = True
                self.washing_completed = False
                self.total_start_time = time.time()
                self.start_step(0)
                print("Hand washing started manually!")
            elif key == ord('n') and self.is_washing and self.step_completed:
                # 다음 단계로 수동 진행
                if self.current_step < len(self.washing_steps) - 1:
                    self.start_step(self.current_step + 1)
                else:
                    # 모든 단계 완료
                    self.washing_completed = True
                    self.is_washing = False
                    print("Hand washing completed!")
                    if 'complete' in self.audio_files:
                        threading.Thread(target=self.play_audio, args=(self.audio_files['complete'],)).start()
            elif key == ord('r'):
                # 리셋
                self.is_washing = False
                self.washing_completed = False
                self.current_step = 0
                self.step_completed = False
                self.movement_counter = 0
                print("System reset!")
            
            # BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # 손 랜드마크 그리기
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 손 씻기 동작 감지
                    landmarks = hand_landmarks.landmark
                    is_washing = self.detect_washing_motion(landmarks)
                    self.current_gesture = self.detect_specific_hand_gesture(landmarks)
                    
                    if is_washing and not self.is_washing and not self.washing_completed:
                        # 손 씻기 시작
                        self.is_washing = True
                        self.washing_completed = False
                        self.total_start_time = time.time()
                        self.start_step(0)
                        print("Hand washing started automatically!")
                    
                    elif self.is_washing and not self.washing_completed:
                        # 현재 시간 계산
                        current_time = time.time()
                        elapsed_time = current_time - self.total_start_time
                        step_time = current_time - self.step_start_time
                        
                        # 단계 완료 확인
                        if step_time >= self.step_duration and not self.step_completed:
                            self.step_completed = True
                            print(f"Step {self.current_step + 1} completed: {self.washing_steps[self.current_step]}")
                            
                            # 자동으로 다음 단계로 진행 (5초 후)
                            if self.current_step < len(self.washing_steps) - 1:
                                threading.Timer(5.0, self.auto_next_step).start()
                            else:
                                # 모든 단계 완료
                                self.washing_completed = True
                                self.is_washing = False
                                print("Hand washing completed!")
                                if 'complete' in self.audio_files:
                                    threading.Thread(target=self.play_audio, args=(self.audio_files['complete'],)).start()
                        
                        # UI 업데이트
                        self.draw_ui(frame, elapsed_time, step_time)
            
            else:
                # 손이 감지되지 않을 때
                if self.is_washing and not self.washing_completed:
                    current_time = time.time()
                    elapsed_time = current_time - self.total_start_time
                    step_time = current_time - self.step_start_time
                    self.draw_ui(frame, elapsed_time, step_time)
                else:
                    self.draw_ui(frame, 0, 0)
            
            # 완료 상태 표시
            if self.washing_completed:
                cv2.putText(frame, "Hand Washing Completed! Press 'r' to restart", (20, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 화면 표시
            cv2.imshow('Hand Washing Guide System', frame)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 임시 파일 정리
        try:
            import shutil
            if os.path.exists("temp_audio"):
                shutil.rmtree("temp_audio")
        except Exception as e:
            print(f"임시 파일 정리 실패: {e}")

if __name__ == "__main__":
    guide = HandWashingGuide()
    guide.run()
