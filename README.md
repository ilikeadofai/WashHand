# 손 씻기 가이드 시스템

웹캠과 컴퓨터 비전 기술을 활용한 실시간 손 씻기 가이드 시스템입니다. 사용자의 손 씻기 과정을 실시간으로 분석하고, 올바른 손 씻기 단계를 안내하며, 권장 시간(30초) 동안 손을 씻는지 측정하여 시각적/청각적 피드백을 제공합니다.

## 주요 기능

- **실시간 손 추적**: MediaPipe를 활용한 정확한 손 랜드마크 추적
- **동작 인식**: 손의 움직임을 분석하여 손 씻기 동작 감지
- **단계별 가이드**: WHO 권장 6단계 손 씻기 과정 안내
- **시간 측정**: 각 단계별 및 전체 시간 측정
- **시각적 피드백**: 진행률 바와 실시간 상태 표시
- **청각적 피드백**: 음성 안내 및 완료 알림

## 설치 방법

### Python 3.10 권장 설치
MediaPipe는 Python 3.10과 가장 호환성이 좋습니다:

1. **Python 버전 확인**: `python_version_check.py` 실행
2. **Python 3.10 설치**: `install_python310.bat` 실행 (권장)
3. **MediaPipe 특별 설치**: `install_mediapipe.bat` 실행
4. **간단 버전 사용**: MediaPipe 없이 기본 기능만 사용

### 자동 설치 (Windows)
1. `install.bat` 파일을 더블클릭하여 실행합니다.
2. MediaPipe 설치 실패 시 `install_mediapipe.bat`를 실행합니다.
3. 설치가 완료될 때까지 기다립니다.

### 수동 설치
1. **Python 3.10 권장** (MediaPipe 최적 호환성)
   - [Python 3.10.9 다운로드](https://www.python.org/downloads/release/python-3109/)
   - 설치 시 'Add Python to PATH' 체크박스 선택

2. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

### 간단 버전 설치 (MediaPipe 없이)
MediaPipe 설치가 어려운 경우:
```bash
pip install -r requirements_simple.txt
python main_simple.py
```

### 시스템 테스트
설치 후 시스템이 제대로 작동하는지 확인하려면:
```bash
python test_system.py
```

## 사용 방법

### 완전 버전 (MediaPipe 포함)
1. 프로그램을 실행합니다:
```bash
python main.py
```

2. 웹캠 앞에 서서 손을 보여주세요.

3. 손을 움직이면 자동으로 손 씻기가 시작됩니다.

### 간단 버전 (MediaPipe 없이)
1. 프로그램을 실행합니다:
```bash
python main_simple.py
```

2. 웹캠 앞에 서서 손을 움직이거나 's' 키를 눌러 시작하세요.

### 머신러닝 기반 동작 인식 (고급 기능)

#### 1단계: 데이터 수집
```bash
python data_collector.py
```
- 각 손 씻기 동작을 10초간 수행하여 데이터 수집
- 1-6 키로 다른 동작 선택
- Space 키로 수집 중지

#### 2단계: 모델 훈련
```bash
python train_model.py
```
- 수집된 데이터로 머신러닝 모델 훈련
- Random Forest, SVM 모델 비교
- 최고 성능 모델 자동 선택

#### 3단계: 향상된 시스템 사용
```bash
python main.py
```
- 훈련된 모델로 정확한 동작 인식
- 실시간 동작 분류 및 피드백

### 공통 기능
- 화면의 안내에 따라 각 단계를 수행하세요:
  - 손바닥 비비기 (10초)
  - 손등 비비기 (10초)
  - 손가락 사이 비비기 (10초)
  - 손가락 끝 비비기 (10초)
  - 엄지손가락 비비기 (10초)
  - 손목 비비기 (10초)

- **컨트롤**:
  - 's': 수동 시작
  - 'n': 다음 단계 진행
  - 'r': 시스템 리셋
  - 'q': 프로그램 종료

## 시스템 요구사항

- Windows 10/11 (Linux/macOS도 지원)
- 웹캠
- **Python 3.10 권장** (MediaPipe 최적 호환성)
- 인터넷 연결 (음성 파일 생성 시 필요)
- 64비트 시스템 권장

## 기술 스택

- **OpenCV**: 실시간 영상 처리
- **MediaPipe**: 손 랜드마크 추적
- **gTTS**: 텍스트-음성 변환
- **NumPy**: 수치 계산
- **PIL**: 이미지 처리

## 프로젝트 구조

```
WashHands/
├── main.py                    # 메인 프로그램 (MediaPipe 포함)
├── main_simple.py             # 간단 버전 (MediaPipe 없이)
├── data_collector.py          # 손 씻기 동작 데이터 수집
├── train_model.py             # 머신러닝 모델 훈련
├── requirements.txt           # 완전 버전 의존성 패키지
├── requirements_simple.txt    # 간단 버전 의존성 패키지
├── requirements_ml.txt        # 머신러닝 의존성 패키지
├── test_system.py             # 시스템 테스트 스크립트
├── python_version_check.py    # Python 버전 확인 스크립트
├── install.bat                # Windows 자동 설치 스크립트
├── install_python310.bat      # Python 3.10 전용 설치 스크립트
├── install_mediapipe.bat      # MediaPipe 특별 설치 스크립트
└── README.md                  # 프로젝트 설명
```

## 개발 배경

코로나19 팬데믹 이후 손 씻기의 중요성이 강조되고 있지만, 많은 사람들이 올바른 손 씻기 방법과 권장 시간을 지키지 않는 경우가 많습니다. 이 프로젝트는 웹캠과 컴퓨터 비전 기술을 활용하여 사용자의 손 씻기 과정을 실시간으로 분석하고, 올바른 손 씻기 습관 형성을 돕는 비대면 시스템을 개발하는 것을 목표로 합니다.

## 향후 개선 계획

- 더 정확한 동작 인식을 위한 머신러닝 모델 적용
- 어린이를 위한 게임 요소 추가
- 비누 거품 감지 기능
- 다양한 언어 지원
- 모바일 앱 버전 개발

## 라이선스

이 프로젝트는 교육 목적으로 개발되었습니다. 