import shutil
import os
import datetime
import hashlib


def calculate_file_hash(file_path):
    block_size = 65536  # 64KB
    hash_algorithm = hashlib.sha256()

    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(block_size), b''):
            hash_algorithm.update(chunk)

    return hash_algorithm.hexdigest()


def backup_files(source_folder, backup_folder):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(backup_folder, exist_ok=True)

    # 이전 백업과 비교할 파일 목록을 저장할 딕셔너리
    previous_files = {}

    # 백업 폴더에 있는 파일들을 검색하여 이전 백업 파일 목록 저장
    for root, dirs, files in os.walk(backup_folder):
        for file in files:
            backup_file = os.path.join(root, file)
            previous_files[file] = calculate_file_hash(backup_file)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)
            backup_filename = f"{timestamp}_{file}"
            backup_path = os.path.join(backup_folder, backup_filename)

            # 이전 백업 파일과 비교하여 해시 값 확인
            previous_file_hash = previous_files.get(file)
            current_file_hash = calculate_file_hash(source_path)

            if previous_file_hash and previous_file_hash == current_file_hash:
                # 파일이 변경되지 않았으면 스킵
                continue

            # 파일이 수정된 경우에만 백업을 수행
            shutil.copy2(source_path, backup_path)
            print(f"백업 완료: {backup_path}")


# 백업할 폴더와 백업 폴더 경로 설정
source_folder = "C:/Users/USER/Desktop/prac"
backup_folder = "C:/Users/USER/Desktop/backup"

# 파일 백업 수행
backup_files(source_folder, backup_folder)
