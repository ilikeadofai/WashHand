# 손 씻기 가이드 시스템 (ML 기반)

웹캠과 머신러닝을 활용한 실시간 손 씻기 동작 인식 및 가이드 시스템입니다. WHO 권장 6단계 손 씻기 과정을 정확한 동작 인식과 함께 안내합니다.

## 🎯 주요 기능

### 🤖 ML 기반 동작 인식
- **MediaPipe**: 실시간 손 랜드마크 추적
- **머신러닝 모델**: 훈련된 모델로 6가지 손 씻기 동작 분류
- **정확한 타이머**: 올바른 동작을 수행할 때만 타이머 진행
- **실시간 피드백**: 동작 일치/불일치 상태 즉시 표시

### 📊 실시간 UI
- **단계별 진행률**: 현재 단계의 완료도 시각화
- **전체 진행률**: 전체 과정에서 올바르게 수행한 시간 합산
- **상태 표시**: "동작 일치" / "동작 불일치" 배지
- **한국어 인터페이스**: 직관적인 한국어 UI

### 🔊 음성 안내
- **단계별 음성**: 각 단계 시작 시 10초 안내 음성
- **완료 알림**: 모든 단계 완료 시 축하 메시지
- **gTTS 기반**: 자연스러운 한국어 음성 생성

## 🏗️ 시스템 구조

### 핵심 컴포넌트
```
main.py                    # 메인 프로그램 (ML 기반)
├── MediaPipe 손 추적      # 실시간 랜드마크 추출
├── ML 모델 추론          # 동작 분류 및 예측
├── 타이머 게이팅         # 올바른 동작 시에만 시간 누적
├── UI 렌더링            # PIL 기반 한국어 인터페이스
└── 음성 안내            # gTTS 기반 음성 출력
```

### 6단계 손 씻기 과정
1. **손바닥 비비기** (10초)
2. **손등 비비기** (10초)
3. **손가락 사이 비비기** (10초)
4. **손가락 끝 비비기** (10초)
5. **엄지손가락 비비기** (10초)
6. **손목 비비기** (10초)

**총 소요 시간**: 60초 (올바른 동작 수행 시)

## 🚀 설치 및 실행

### 필수 요구사항
- **Python 3.10+** (MediaPipe 최적 호환성)
- **웹캠** (실시간 영상 입력)
- **Windows 10/11** (권장, Linux/macOS도 지원)

### 패키지 설치
```bash
# 기본 패키지
pip install opencv-python mediapipe numpy

# ML 및 음성 패키지
pip install joblib gtts Pillow

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

### 실행 방법
```bash
# ML 기반 손 씻기 가이드 실행
python main.py
```

## 🎮 사용법

### 기본 조작
- **시작**: `s` 키를 눌러 손 씻기 시작
- **종료**: `q` 키를 눌러 프로그램 종료
- **자동 진행**: 올바른 동작 수행 시 자동으로 다음 단계 진행

### 동작 인식 원리
1. **손 감지**: MediaPipe가 웹캠에서 손을 실시간 감지
2. **특징 추출**: 손 랜드마크에서 63개 특징 벡터 추출
3. **ML 예측**: 훈련된 모델로 현재 동작 분류
4. **안정화**: 최근 8프레임 중 5프레임 이상 일치 시 "올바른 동작" 인정
5. **타이머 게이팅**: 올바른 동작일 때만 시간 누적

### UI 해석
- **초록색 진행바**: 올바른 동작 수행 중 (시간 증가)
- **빨간색 배지**: 잘못된 동작 (시간 정지)
- **단계 진행률**: 현재 단계 완료도 (0-10초)
- **전체 진행률**: 전체 과정 완료도 (0-60초)

## 🔧 고급 기능

### ML 모델 관리
```bash
# 모델 자동 로드
trained_models/*.joblib  # 최신 모델 자동 선택

# 모델 없을 경우
# 1. data_collector.py로 데이터 수집
# 2. train_model.py로 모델 훈련
# 3. trained_models/에 저장
```

### 음성 설정
- **언어**: 한국어 (ko)
- **생성 위치**: `temp_audio/` 폴더
- **자동 정리**: 프로그램 종료 시 임시 파일 삭제

### 예측 안정화
- **히스토리 길이**: 최근 8프레임
- **과반수 기준**: 5프레임 이상 일치
- **확률 임계값**: 0.35 (predict_proba 지원 시)

## 📁 프로젝트 구조

```
WashHands/
├── main.py                    # 메인 프로그램 (ML 기반)
├── main_simple.py             # 간단 버전 (움직임 감지)
├── data_collector.py          # ML 데이터 수집
├── train_model.py             # ML 모델 훈련
├── test_ui_states.py          # UI 테스트 (카메라 없이)
├── requirements.txt           # 필수 패키지 목록
├── trained_models/            # 훈련된 ML 모델 저장
├── temp_audio/                # 음성 파일 임시 저장
└── README.md                  # 프로젝트 문서
```

## 🧪 테스트 및 개발

### UI 테스트
```bash
# 카메라 없이 UI 상태 테스트
python test_ui_states.py

# 컨트롤:
# m: 일치/불일치 토글
# n: 다음 단계
# a: 자동 전환
# q: 종료
```

### ML 모델 개발
```bash
# 1. 데이터 수집
python data_collector.py

# 2. 모델 훈련
python train_model.py

# 3. 메인 프로그램 실행
python main.py
```

## 🔍 기술 상세

### 특징 추출 (63차원)
```python
def extract_hand_features(landmarks):
    features = []
    # 모든 랜드마크 좌표 (21개 × 3 = 63)
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features
```

### 예측 안정화 알고리즘
```python
# 최근 8프레임 히스토리
pred_hist = deque(maxlen=8)

# 과반수 기준 (5/8)
need_majority = 5

# 현재 단계와 일치하는 프레임 수 계산
cnt = sum(1 for x in pred_hist if x == need_key)
return cnt >= need_majority
```

### 타이머 게이팅
```python
# 올바른 동작일 때만 시간 누적
if matched:
    self.step_active_time += dt
    self.total_active_time += dt
```

## 🐛 문제 해결

### 일반적인 문제
1. **웹캠 인식 안됨**: 웹캠 권한 및 드라이버 확인
2. **ML 모델 없음**: `trained_models/` 폴더에 .joblib 파일 필요
3. **음성 안됨**: 인터넷 연결 및 gTTS 설치 확인
4. **폰트 깨짐**: Windows 환경에서 malgun.ttf 확인

### 성능 최적화
- **해상도 조정**: 웹캠 해상도 낮춰서 성능 향상
- **프레임 스킵**: 필요시 프레임 처리 간격 조정
- **모델 경량화**: 더 작은 ML 모델 사용

## 📄 라이선스

이 프로젝트는 Apache License 2.0 하에 배포됩니다.

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 문의

프로젝트 관련 문의사항이나 버그 리포트는 GitHub Issues를 통해 제출해주세요.

---

**개발 목표**: 정확한 손 씻기 습관 형성을 위한 AI 기반 교육 시스템
**적용 분야**: 의료기관, 교육기관, 공공시설, 개인 위생 교육