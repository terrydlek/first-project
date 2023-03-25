'''
1 x 1 크기의 칸들로 이루어진 직사각형 격자 형태의 미로에서 탈출하려고 합니다.
각 칸은 통로 또는 벽으로 구성되어 있으며, 벽으로 된 칸은 지나갈 수 없고 통로로 된 칸으로만 이동할 수 있습니다.
통로들 중 한 칸에는 미로를 빠져나가는 문이 있는데, 이 문은 레버를 당겨서만 열 수 있습니다. 레버 또한 통로들 중 한 칸에 있습니다.
따라서, 출발 지점에서 먼저 레버가 있는 칸으로 이동하여 레버를 당긴 후 미로를 빠져나가는 문이 있는 칸으로 이동하면 됩니다.
이때 아직 레버를 당기지 않았더라도 출구가 있는 칸을 지나갈 수 있습니다.
미로에서 한 칸을 이동하는데 1초가 걸린다고 할 때, 최대한 빠르게 미로를 빠져나가는데 걸리는 시간을 구하려 합니다.
미로를 나타낸 문자열 배열 maps가 매개변수로 주어질 때,
미로를 탈출하는데 필요한 최소 시간을 return 하는 solution 함수를 완성해주세요. 만약, 탈출할 수 없다면 -1을 return 해주세요.
제한사항
5 ≤ maps의 길이 ≤ 100
5 ≤ maps[i]의 길이 ≤ 100
maps[i]는 다음 5개의 문자들로만 이루어져 있습니다.
S : 시작 지점
E : 출구
L : 레버
O : 통로
X : 벽
시작 지점과 출구, 레버는 항상 다른 곳에 존재하며 한 개씩만 존재합니다.
출구는 레버가 당겨지지 않아도 지나갈 수 있으며, 모든 통로, 출구, 레버, 시작점은 여러 번 지나갈 수 있습니다.
'''
from collections import deque
row = int(input())
maps = [list(map(str, input().split())) for _ in range(row)]


def solution(maps):
    maps = [list(i) for i in maps]
    
    def bfs(start, end):
        q = deque()
        y, x = start
        endy, endx = end
        time = 0
        q.append((y, x, time))
        visited = [["U"] * len(maps[0]) for _ in range(len(maps))]
        while q:
            y, x, time = q.popleft()
            
            if y == endy and x == endx:
                return time
            
            if y + 1 < len(maps) and visited[y + 1][x] != "V" and maps[y + 1][x] != "X":
                q.append((y + 1, x, time + 1))
                visited[y + 1][x] = "V"
            if y - 1 >= 0 and visited[y - 1][x] !=  "V" and maps[y - 1][x] != "X":
                q.append((y - 1, x, time + 1))
                visited[y - 1][x] = "V"
            if x + 1 < len(maps[0]) and visited[y][x + 1] != "V" and maps[y][x + 1] != "X":
                q.append((y, x + 1, time + 1))
                visited[y][x + 1] = "V"
            if x - 1 >= 0 and visited[y][x - 1] != "V" and maps[y][x - 1] != "X":
                q.append((y, x - 1, time + 1))
                visited[y][x - 1] = "V"
        return -1
    
    start = 0, 0
    lever = 0, 0
    exit = 0, 0

    for i in range(len(maps)):
        for j in range(len(maps[0])):
            if maps[i][j] == "S":
                start = i, j
            if maps[i][j] == "L":
                lever = i, j
            if maps[i][j] == "E":
                exit = i, j
                
    one = bfs(start, lever)
    two = bfs(lever, exit)
    if one == -1 or two == -1:
        return -1
    return one + two


print(solution(maps))
