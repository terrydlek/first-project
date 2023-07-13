from distutils.dir_util import copy_tree

original = r'C:\Users\USER\PycharmProjects\pythonProject3\전산 프로젝트\파일 자동 백업 프로그램\원본'
backup = r'C:\Users\USER\PycharmProjects\pythonProject3\전산 프로젝트\파일 자동 백업 프로그램\백업'

result = copy_tree(original, backup, update=1) # 바뀐 파일만 백업함(update = 1)
print(result)

