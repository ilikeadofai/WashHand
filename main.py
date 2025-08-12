import cv2
import mediapipe as mp
import numpy as np
import joblib
import glob
import os
import time
import threading
from collections import deque
from PIL import Image, ImageDraw, ImageFont

STEP_SECONDS = 10         # 단계별 목표 10초 (표시/음성/실제 측정 모두 동일)
TOTAL_STEPS = 6
TOTAL_SECONDS = STEP_SECONDS * TOTAL_STEPS  # 60초

# MediaPipe 특징 추출 (기존 main.py와 동일 형태) 
def extract_hand_features(landmarks):
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])
    finger_tips = [8, 12, 16, 20]
    for tip in finger_tips:
        features.extend([landmarks[tip].x, landmarks[tip].y, landmarks[tip].z])
    finger_mids = [6, 10, 14, 18]
    for mid in finger_mids:
        features.extend([landmarks[mid].x, landmarks[mid].y, landmarks[mid].z])
    thumb_tip = landmarks[4]; thumb_ip = landmarks[3]
    features.extend([thumb_tip.x, thumb_tip.y, thumb_tip.z])
    features.extend([thumb_ip.x, thumb_ip.y, thumb_ip.z])
    wrist = landmarks[0]
    features.extend([wrist.x, wrist.y, wrist.z])
    return features

class HandWashingGuideML:
    def __init__(self):
        # 1) MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.5
        )

        # 2) 모델 자동 로드 (trained_models/*.joblib 중 최신)
        model_files = glob.glob('trained_models/*.joblib')
        if model_files:
            latest = max(model_files, key=os.path.getctime)
            self.model = joblib.load(latest)
            print(f"[INFO] Loaded model: {latest}")
        else:
            self.model = None
            print("[WARN] No trained model found. Please train and put a .joblib into trained_models/")

        # 3) 단계 정의 (라벨 키 ↔ 한글명 ↔ 안내문)
        self.step_keys = [
            'palm_rubbing', 'back_rubbing', 'finger_gaps',
            'finger_tips', 'thumb_rubbing', 'wrist_rubbing'
        ]
        self.step_names = {
            'palm_rubbing': '손바닥 비비기',
            'back_rubbing': '손등 비비기',
            'finger_gaps': '손가락 사이 비비기',
            'finger_tips': '손가락 끝 비비기',
            'thumb_rubbing': '엄지손가락 비비기',
            'wrist_rubbing': '손목 비비기'
        }
        self.current_step = 0
        self.step_active_time = 0.0      # 현재 단계에서 "맞게 수행한" 누적 시간(초)
        self.total_active_time = 0.0     # 전체에서 "맞게 수행한" 누적 시간(초)
        self.running = False
        self.last_time = None

        # 4) 예측 안정화(스무딩)
        self.pred_hist = deque(maxlen=8)   # 최근 8프레임
        self.need_majority = 5             # 그중 >=5 프레임이 현재 단계 라벨이면 "맞게 수행 중"으로 인정
        self.proba_thresh = 0.35           # predict_proba 있으면 사용 (없으면 무시)

        # 5) 오디오(10초 문구로 통일)
        self.audio = {}
        self._prepare_audio()

    def _prepare_audio(self):
        try:
            from gtts import gTTS
            os.makedirs("temp_audio", exist_ok=True)
            for i, key in enumerate(self.step_keys):
                text = f"{self.step_names[key]}를 시작합니다. 10초간 계속해주세요."
                path = f"temp_audio/step_{i}.mp3"
                if not os.path.exists(path):
                    gTTS(text=text, lang='ko').save(path)
                self.audio[i] = path
            done_path = "temp_audio/complete.mp3"
            if not os.path.exists(done_path):
                gTTS(text="손 씻기가 완료되었습니다! 잘 하셨습니다.", lang='ko').save(done_path)
            self.audio['complete'] = done_path
        except Exception as e:
            print(f"[WARN] TTS 생성 실패: {e} (인터넷 연결 또는 gTTS 확인)")

    def _play_audio_async(self, path):
        if not path or not os.path.exists(path):
            return
        def _player():
            try:
                # OS 기본 플레이어로 열기 (간단)
                import platform, subprocess
                if platform.system() == "Windows":
                    os.startfile(os.path.abspath(path))
                elif platform.system() == "Darwin":
                    subprocess.call(["open", path])
                else:
                    subprocess.call(["xdg-open", path])
            except Exception as e:
                print(f"[WARN] 오디오 재생 실패: {e}")
        threading.Thread(target=_player, daemon=True).start()

    def _predict_label(self, landmarks):
        if self.model is None or landmarks is None:
            return None, None
        feats = extract_hand_features(landmarks)
        label = None
        proba = None
        try:
            # RandomForest 등은 predict_proba 지원
            label = self.model.predict([feats])[0]
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba([feats])[0]
                # 클래스 인덱스 찾기
                try:
                    classes = list(self.model.classes_)
                    if label in classes:
                        proba = float(probs[classes.index(label)])
                except Exception:
                    proba = None
        except Exception as e:
            print(f"[WARN] 예측 실패: {e}")
        return label, proba

    def _matched_current_step(self, label, proba):
        """현재 단계와 라벨 일치 + (있다면) 확률 임계 통과 + 히스토리 과반수"""
        if label is None:
            self.pred_hist.append("_")
            return False
        need_key = self.step_keys[self.current_step]
        ok_label = (label == need_key)
        ok_proba = (proba is None) or (proba >= self.proba_thresh)
        self.pred_hist.append(label if ok_label and ok_proba else "_")
        # 최근 히스토리 중 현재 단계 라벨 개수
        cnt = sum(1 for x in self.pred_hist if x == need_key)
        return cnt >= self.need_majority

    def start(self):
        self.running = True
        self.current_step = 0
        self.step_active_time = 0.0
        self.total_active_time = 0.0
        self.last_time = time.time()
        if 0 in self.audio:
            self._play_audio_async(self.audio[0])

    def _advance_step(self):
        self.current_step += 1
        self.step_active_time = 0.0
        self.pred_hist.clear()
        if self.current_step < TOTAL_STEPS and self.current_step in self.audio:
            self._play_audio_async(self.audio[self.current_step])

    def _draw_ui(self, frame, matched):
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 32)
            font_body  = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
        except:
            font_title = ImageFont.load_default(); font_body = ImageFont.load_default()

        # 헤더
        draw.text((20, 20), "손 씻기 가이드 (ML)", font=font_title, fill=(255,255,255,255))

        # 현재 단계
        step_txt = f"현재 단계: {self.step_names[self.step_keys[self.current_step]] if self.current_step<TOTAL_STEPS else '완료'}"
        draw.text((20, 60), step_txt, font=font_body, fill=(0,255,255,255))

        # 일치 여부 배지
        badge = "동작 일치" if matched else "동작 불일치"
        badge_color = (0, 255, 0, 255) if matched else (255, 80, 80, 255)
        draw.text((20, 90), badge, font=font_body, fill=badge_color)

        # 단계 진행바
        step_progress = 1.0 if self.current_step >= TOTAL_STEPS else min(1.0, self.step_active_time / STEP_SECONDS)
        x0, y0, w, h = 20, 130, 400, 18
        draw.rectangle([x0, y0, x0+w, y0+h], fill=(60,60,60,200))
        draw.rectangle([x0, y0, x0+int(w*step_progress), y0+h], fill=(0,200,0,255))
        draw.text((x0+w+10, y0-2), f"{self.step_active_time:4.1f}s / {STEP_SECONDS}s", font=font_body, fill=(255,255,255,255))

        # 전체 진행바(맞게 수행한 시간 합산)
        total_progress = min(1.0, self.total_active_time / TOTAL_SECONDS)
        X0, Y0, W, H = 20, 160, 500, 18
        draw.rectangle([X0, Y0, X0+W, Y0+H], fill=(60,60,60,200))
        draw.rectangle([X0, Y0, X0+int(W*total_progress), Y0+H], fill=(0,255,255,255))
        draw.text((X0+W+10, Y0-2), f"{self.total_active_time:4.1f}s / {TOTAL_SECONDS}s", font=font_body, fill=(255,255,255,255))

        # 적용
        frame[:,:,:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return

        print("ML 손 씻기 가이드 시작 (q: 종료, s: 시작/리셋)")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.start()

            # 인식/진행
            matched = False
            if self.running and self.current_step < TOTAL_STEPS and self.model is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                    label, proba = self._predict_label(hand.landmark)
                    matched = self._matched_current_step(label, proba)
                else:
                    self.pred_hist.append("_")

                # 시간 누적은 "matched일 때만"
                now = time.time()
                dt = max(0.0, now - (self.last_time or now))
                self.last_time = now

                if matched:
                    self.step_active_time += dt
                    self.total_active_time += dt

                # 단계 완료 판정 (딱 10초 기준)
                if self.step_active_time >= STEP_SECONDS - 1e-6:
                    self._advance_step()
                    if self.current_step >= TOTAL_STEPS:
                        # 전체 완료
                        if 'complete' in self.audio:
                            self._play_audio_async(self.audio['complete'])
                        self.running = False  # 자동 정지

            # UI
            self._draw_ui(frame, matched)
            cv2.imshow('손 씻기 가이드 (ML)', frame)

        cap.release()
        cv2.destroyAllWindows()

        # 임시 오디오 정리 (선택)
        try:
            import shutil
            if os.path.exists("temp_audio"):
                shutil.rmtree("temp_audio")
        except Exception as e:
            print(f"[WARN] 임시 파일 정리 실패: {e}")

if __name__ == "__main__":
    HandWashingGuideML().run()
