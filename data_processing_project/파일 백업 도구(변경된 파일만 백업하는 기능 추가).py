import shutil
import os
import datetime


def backup_files(source_folder, backup_folder):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(backup_folder, exist_ok=True)

    # 이전 백업과 비교할 파일 목록을 저장할 딕셔너리
    previous_files = {}

    # 백업 폴더에 있는 파일들을 검색하여 이전 백업 파일 목록 저장
    for root, dirs, files in os.walk(backup_folder):
        for file in files:
            backup_file = os.path.join(root, file)
            previous_files[file] = os.path.getmtime(backup_file)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)
            backup_filename = f"{timestamp}_{file}"
            backup_path = os.path.join(backup_folder, backup_filename)

            # 이전 백업 파일과 비교하여 수정 일자 확인
            previous_file_mtime = previous_files.get(file)
            if previous_file_mtime and os.path.getmtime(source_path) <= previous_file_mtime:
                continue  # 파일이 변경되지 않았으면 스킵

            shutil.copy2(source_path, backup_path)
            print(f"백업 완료: {backup_path}")


# 백업할 폴더와 백업 폴더 경로 설정
source_folder = "C:/Users/USER/Desktop/prac"
backup_folder = "C:/Users/USER/Desktop/backup"

# 파일 백업 수행
backup_files(source_folder, backup_folder)
