import cv2
import numpy as np
import time
import threading
import os
import platform

class SimpleHandWashingGuide:
    def __init__(self):
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
        self.step_duration = 5  # 각 단계별 권장 시간 (초)
        self.total_duration = 30  # 전체 권장 시간 (초)
        
        # 움직임 감지 변수
        self.prev_frame = None
        self.movement_threshold = 30
        self.movement_counter = 0
        self.movement_threshold_count = 10
        
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
                from gtts import gTTS
                tts = gTTS(text=f"{step}를 시작합니다. 5초간 계속해주세요.", lang='ko')
                audio_path = f"temp_audio/step_{i}.mp3"
                tts.save(audio_path)
                self.audio_files[i] = audio_path
            except Exception as e:
                print(f"음성 파일 생성 실패: {e}")
        
        # 완료 메시지
        try:
            from gtts import gTTS
            tts = gTTS(text="손 씻기가 완료되었습니다! 잘 하셨습니다.", lang='ko')
            audio_path = "temp_audio/complete.mp3"
            tts.save(audio_path)
            self.audio_files['complete'] = audio_path
        except Exception as e:
            print(f"완료 음성 파일 생성 실패: {e}")
    
    def detect_motion(self, frame):
        """프레임 간 움직임을 감지합니다."""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False
        
        # 현재 프레임을 그레이스케일로 변환
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 프레임 간 차이 계산
        frame_diff = cv2.absdiff(self.prev_frame, current_frame)
        
        # 임계값 적용
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # 움직임 영역 계산
        motion_pixels = np.sum(thresh > 0)
        
        # 이전 프레임 업데이트
        self.prev_frame = current_frame
        
        # 움직임 감지
        if motion_pixels > self.movement_threshold * 1000:  # 임계값 조정
            self.movement_counter += 1
        else:
            self.movement_counter = max(0, self.movement_counter - 1)
        
        return self.movement_counter >= self.movement_threshold_count
    
    def draw_ui(self, frame, elapsed_time, step_time):
        """UI를 그립니다."""
        height, width = frame.shape[:2]
        
        # 배경 오버레이
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 제목
        cv2.putText(frame, "손 씻기 가이드 시스템 (간단 버전)", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 현재 단계
        if self.is_washing:
            current_step_text = f"현재 단계: {self.washing_steps[self.current_step]}"
            cv2.putText(frame, current_step_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 진행률 표시
            progress = min(1.0, step_time / self.step_duration)
            bar_width = int(300 * progress)
            cv2.rectangle(frame, (20, 80), (320, 90), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 80), (20 + bar_width, 90), (0, 255, 0), -1)
            
            # 시간 표시
            time_text = f"단계 시간: {step_time:.1f}s / {self.step_duration}s"
            cv2.putText(frame, time_text, (330, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 전체 시간
            total_time_text = f"전체 시간: {elapsed_time:.1f}s / {self.total_duration}s"
            cv2.putText(frame, total_time_text, (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 전체 진행률
            total_progress = min(1.0, elapsed_time / self.total_duration)
            total_bar_width = int(400 * total_progress)
            cv2.rectangle(frame, (20, 140), (420, 150), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 140), (20 + total_bar_width, 150), (0, 255, 255), -1)
        else:
            cv2.putText(frame, "손을 움직여주세요", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "또는 's' 키를 눌러 시작", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def play_audio(self, audio_path):
        """음성을 재생합니다."""
        try:
            # 간단한 방법으로 음성 파일 실행
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
        
        print(f"단계 {step_index + 1} 시작: {self.washing_steps[step_index]}")
        
        # 음성 안내 재생
        if step_index in self.audio_files:
            threading.Thread(target=self.play_audio, args=(self.audio_files[step_index],)).start()
    
    def run(self):
        """메인 실행 루프"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return
        
        print("간단한 손 씻기 가이드 시스템을 시작합니다.")
        print("손을 움직이거나 's' 키를 눌러 시작하세요.")
        print("'q' 키를 누르면 종료됩니다.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 움직임 감지
            is_moving = self.detect_motion(frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not self.is_washing:
                # 수동 시작
                self.is_washing = True
                self.total_start_time = time.time()
                self.start_step(0)
                print("손 씻기를 시작합니다!")
            
            # 움직임으로 자동 시작
            if is_moving and not self.is_washing:
                self.is_washing = True
                self.total_start_time = time.time()
                self.start_step(0)
                print("손 씻기를 시작합니다!")
            
            # 시간 계산 및 UI 업데이트
            if self.is_washing:
                current_time = time.time()
                elapsed_time = current_time - self.total_start_time
                step_time = current_time - self.step_start_time
                
                # 단계 완료 확인
                if step_time >= self.step_duration and not self.step_completed:
                    self.step_completed = True
                    print(f"단계 {self.current_step + 1} 완료: {self.washing_steps[self.current_step]}")
                    
                    # 다음 단계로 진행
                    if self.current_step < len(self.washing_steps) - 1:
                        self.start_step(self.current_step + 1)
                    else:
                        # 모든 단계 완료
                        if elapsed_time >= self.total_duration:
                            print("손 씻기 완료!")
                            if 'complete' in self.audio_files:
                                threading.Thread(target=self.play_audio, args=(self.audio_files['complete'],)).start()
                            self.is_washing = False
                            self.current_step = 0
                
                # UI 업데이트
                self.draw_ui(frame, elapsed_time, step_time)
            else:
                self.draw_ui(frame, 0, 0)
            
            # 화면 표시
            cv2.imshow('손 씻기 가이드 시스템 (간단 버전)', frame)
        
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
    guide = SimpleHandWashingGuide()
    guide.run() 