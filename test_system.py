#!/usr/bin/env python3
"""
손 씻기 가이드 시스템 테스트 스크립트
"""

import sys
import subprocess
import importlib

def check_package(package_name):
    """패키지가 설치되어 있는지 확인합니다."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """패키지를 설치합니다."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("손 씻기 가이드 시스템 테스트를 시작합니다...")
    print("=" * 50)
    
    # 필요한 패키지 목록
    required_packages = [
        "opencv-python",
        "mediapipe", 
        "numpy",
        "gtts",
        "Pillow"
    ]
    
    print("필요한 패키지들을 확인하는 중...")
    
    missing_packages = []
    for package in required_packages:
        if check_package(package.replace("-", "_")):
            print(f"✓ {package} - 설치됨")
        else:
            print(f"✗ {package} - 설치되지 않음")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n설치되지 않은 패키지들이 있습니다.")
        response = input("자동으로 설치하시겠습니까? (y/n): ")
        
        if response.lower() == 'y':
            print("\n패키지 설치 중...")
            for package in missing_packages:
                print(f"{package} 설치 중...")
                if install_package(package):
                    print(f"✓ {package} 설치 완료")
                else:
                    print(f"✗ {package} 설치 실패")
                    return False
        else:
            print("패키지 설치를 취소했습니다.")
            print("requirements.txt를 참고하여 수동으로 설치해주세요.")
            return False
    
    print("\n웹캠 테스트를 시작합니다...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ 웹캠을 열 수 없습니다.")
            print("웹캠이 연결되어 있고 다른 프로그램에서 사용 중이 아닌지 확인해주세요.")
            return False
        
        print("✓ 웹캠 연결 성공")
        
        # 몇 프레임을 캡처하여 웹캠이 작동하는지 확인
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"✓ 프레임 {i+1} 캡처 성공")
            else:
                print(f"✗ 프레임 {i+1} 캡처 실패")
                cap.release()
                return False
        
        cap.release()
        print("✓ 웹캠 테스트 완료")
        
    except Exception as e:
        print(f"✗ 웹캠 테스트 실패: {e}")
        return False
    
    print("\nMediaPipe 테스트를 시작합니다...")
    
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe 초기화 성공")
        
    except Exception as e:
        print(f"✗ MediaPipe 테스트 실패: {e}")
        return False
    
    print("\n음성 생성 테스트를 시작합니다...")
    
    try:
        from gtts import gTTS
        import os
        
        # 임시 음성 파일 생성
        tts = gTTS(text="테스트 음성입니다.", lang='ko')
        test_audio_path = "test_audio.mp3"
        tts.save(test_audio_path)
        
        if os.path.exists(test_audio_path):
            print("✓ 음성 파일 생성 성공")
            # 테스트 파일 삭제
            os.remove(test_audio_path)
        else:
            print("✗ 음성 파일 생성 실패")
            return False
            
    except Exception as e:
        print(f"✗ 음성 생성 테스트 실패: {e}")
        print("인터넷 연결을 확인해주세요.")
        return False
    
    print("\n" + "=" * 50)
    print("✓ 모든 테스트가 성공적으로 완료되었습니다!")
    print("이제 main.py를 실행하여 손 씻기 가이드 시스템을 사용할 수 있습니다.")
    print("\n실행 방법:")
    print("python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n테스트가 실패했습니다. 위의 오류 메시지를 확인하고 문제를 해결한 후 다시 시도해주세요.")
        sys.exit(1) 