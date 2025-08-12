import os

# 원하는 세션 폴더명 리스트 (필요에 따라 수정)
SESSIONS = [
    'session1',
    'session2',
    'Interface_number_1',
]

DATASET_DIR = 'dataset'
DUMMY_VIDEO = b''  # 빈 파일로 생성
DUMMY_ANNOTATION = '{"annotations": []}'

def make_structure():
    os.makedirs(DATASET_DIR, exist_ok=True)
    for session in SESSIONS:
        session_dir = os.path.join(DATASET_DIR, session)
        os.makedirs(session_dir, exist_ok=True)
        video_path = os.path.join(session_dir, 'video.mp4')
        annotation_path = os.path.join(session_dir, 'annotation.json')
        # 더미 video.mp4 생성 (없을 때만)
        if not os.path.exists(video_path):
            with open(video_path, 'wb') as f:
                f.write(DUMMY_VIDEO)
        # 더미 annotation.json 생성 (없을 때만)
        if not os.path.exists(annotation_path):
            with open(annotation_path, 'w', encoding='utf-8') as f:
                f.write(DUMMY_ANNOTATION)
        print(f"구성 완료: {session_dir}")

if __name__ == "__main__":
    make_structure()
    print("\n폴더/파일 구조가 준비되었습니다. 실제 데이터로 video.mp4, annotation.json을 교체하세요.") 