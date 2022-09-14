n, m = map(int, input().split()) #세로크기 N, 가로크기 M
d = [[0] * m for _ in range(n)] #다녀온 위치를 저장하기 위해 만듦
x, y, direction = map(int, input().split()) #현재 위치, 방향 입력
d[x][y] = 1 #현재 좌표 방문처리

array = [] #게임판을  만들기 위해
for i in range(n): #게임판의 정보 입력
    array.append(list(map(int, input().split())))

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

#왼쪽으로 회전
def turn_left():
    global direction
    direction -= 1
    if direction == -1:
        direction = 3

count = 1
turn_time = 0
while True:
    #왼쪽으로 회전
    turn_left()
    nx = x + dx[direction]
    ny = y + dy[direction]
    #회전한 이후 정면에 가보지 않는 칸이 존재하는 경우 이동
    if d[nx][ny] == 0 and array[nx][ny] == 0:
        d[nx][ny] = 1
        x = nx
        y = ny
        count += 1
        turn_time = 0
        continue
    else:
        turn_time += 1
    #회전한 이후 정면에 가보지 않은 칸이 없거나 바다인 경우
    if turn_time == 4:
        nx = x - dx[direction]
        ny = y - dy[direction]
        #뒤로 갈 수 있다면 이동하기
        if array[nx][ny] == 0:
            x = nx
            y = ny
        #뒤가 바다로 막혀있는 경우
        else:
            break
        turn_time = 0
print(count)
