import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

def collect_gesture_data(gesture_name, num_samples, frames_per_sample, output_dir):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(0)
    os.makedirs(output_dir, exist_ok=True)
    sample_count = 0
    print(f"\n제스처: {gesture_name} | 샘플 수집: {num_samples}개 | 샘플당 프레임: {frames_per_sample}")
    print("스페이스바: 샘플 수집 시작/정지, q: 종료")
    while sample_count < num_samples:
        frames = []
        collecting = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            show_text = f"{gesture_name} 샘플 {sample_count+1}/{num_samples}"
            if collecting:
                show_text += f" | 녹화 중: {len(frames)}/{frames_per_sample}"
            cv2.putText(frame, show_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('데이터 수집', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                return
            if not collecting and key == ord(' '):
                collecting = True
                frames = []
                print(f"샘플 {sample_count+1} 녹화 시작...")
            if collecting:
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0].landmark
                    landmarks = [v for point in lm for v in (point.x, point.y, point.z)]
                else:
                    landmarks = [0.0]*63
                frames.append(landmarks)
                if len(frames) >= frames_per_sample:
                    # 샘플 저장
                    sample = {
                        'gesture': gesture_name,
                        'frames': frames
                    }
                    out_path = os.path.join(output_dir, f"{gesture_name}_{sample_count+1}.pkl")
                    with open(out_path, 'wb') as f:
                        pickle.dump(sample, f)
                    print(f"샘플 {sample_count+1} 저장 완료: {out_path}")
                    sample_count += 1
                    break
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n데이터 수집 완료!")

if __name__ == "__main__":
    gesture = input("수집할 제스처 이름을 입력하세요: ")
    num_samples = int(input("샘플 개수(몇 번 반복): "))
    frames_per_sample = int(input("샘플당 프레임 수(길이): "))
    output_dir = f"hand_washing_data/{gesture}"
    collect_gesture_data(gesture, num_samples, frames_per_sample, output_dir) 