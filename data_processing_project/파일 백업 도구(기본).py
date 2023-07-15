import shutil
import os
import datetime


def backup_files(source_folder, backup_folder):
    # 현재 날짜와 시간을 이용하여 백업 파일명 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # 백업 폴더 생성 (이미 존재하는 경우 무시)
    os.makedirs(backup_folder, exist_ok=True)

    # 소스 폴더의 파일들을 순회하며 백업 수행
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 소스 파일 경로
            source_path = os.path.join(root, file)

            # 백업 파일 경로 (원본 파일명 앞에 날짜와 시간 추가)
            backup_filename = f"{timestamp}_{file}"
            backup_path = os.path.join(backup_folder, backup_filename)

            # 파일 복사
            shutil.copy2(source_path, backup_path)

            # 복사된 파일 경로 출력
            print(f"백업 완료: {backup_path}")


# 백업할 폴더와 백업 폴더 경로 설정
source_folder = "C:/Users/USER/Desktop/삼각함수/과제"
backup_folder = "C:/Users/USER/Desktop/backup"

# 파일 백업 수행
backup_files(source_folder, backup_folder)
