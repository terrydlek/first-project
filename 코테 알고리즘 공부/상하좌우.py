'''N = int(input("공간의 크기: "))
A = list(input("계획서 내용: ").split())
now = [1, 1]
R = [now[0], now[1] + 1]
L = [now[0], now[1] - 1]
U = [now[0] - 1, now[1]]
D = [now[0] + 1, now[1]]
for i in A:
    if i == "R":
        now = [now[0], now[1] + 1]
        if now[0] > N or now[0] < 1 or now[1] > N or now[1] < 0:
            now = [now[0], now[1] - 1]
    elif i == "L":
        now = [now[0], now[1] - 1]
        if now[0] > N or now[0] < 1 or now[1] > N or now[1] < 0:
            now = [now[0], now[1] + 1]
    elif i == "U":
        now = [now[0] - 1, now[1]]
        if now[0] > N or now[0] < 1 or now[1] > N or now[1] < 0:
            now = [now[0] + 1, now[1]]
    elif i == "D":
        now = [now[0] + 1, now[1]]
        if now[0] > N or now[0] < 1 or now[1] > N or now[1] < 0:
            now = [now[0] - 1, now[1]]

print(now[0], now[1])'''
#or
n = int(input()) #n을 입력받기
x, y = 1, 1
plans = input().split()
#L, R, U, D에 따른 이동 방향
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]
move_types = ['L', 'R', 'U', 'D']
#이동 계획을 하나씩 확인
for plan in plans:
    #이동 후 좌표 구하기
    for i in range(len(move_types)):
        if plan == move_types[i]:
            nx = x + dx[i]
            ny = y + dy[i]
    #공간을 벗어나는 경우 무시
    if nx < 1 or ny < 1 or nx > n or ny > n:
        continue
    #이동 수행
    x, y = nx, ny
print(x, y)
