# batch 파일을 만든 후 백업 알람 프로그램 실행 명령어를 batch 파일에 붙여넣기.
# 작업 스케줄러, 기본작업 만들기 들어간 후 backup파일 업로드, 시간 설정하면 설정한 시간에 자동으로 프로그램 실행!
from distutils.dir_util import copy_tree
from win10toast import ToastNotifier

toaster = ToastNotifier()

original = r'C:\Users\USER\PycharmProjects\pythonProject3\전산 프로젝트\파일 자동 백업 프로그램\원본'
backup = r'C:\Users\USER\PycharmProjects\pythonProject3\전산 프로젝트\파일 자동 백업 프로그램\백업'
result = copy_tree(original, backup, update=1) # 바뀐 파일만 백업함(update = 1)

toaster.show_toast("백업이 완료되었습니다.", original + ">>>" + backup, duration=10)
