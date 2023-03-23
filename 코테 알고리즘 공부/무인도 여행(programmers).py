'''
메리는 여름을 맞아 무인도로 여행을 가기 위해 지도를 보고 있습니다.
지도에는 바다와 무인도들에 대한 정보가 표시돼 있습니다.
지도는 1 x 1크기의 사각형들로 이루어진 직사각형 격자 형태이며, 격자의 각 칸에는 'X' 또는 1에서 9 사이의 자연수가 적혀있습니다.
지도의 'X'는 바다를 나타내며, 숫자는 무인도를 나타냅니다.
이때, 상, 하, 좌, 우로 연결되는 땅들은 하나의 무인도를 이룹니다.
지도의 각 칸에 적힌 숫자는 식량을 나타내는데, 상, 하, 좌, 우로 연결되는 칸에 적힌 숫자를 모두 합한 값은 해당 무인도에서 최대 며칠동안 머물 수 있는지를 나타냅니다.
어떤 섬으로 놀러 갈지 못 정한 메리는 우선 각 섬에서 최대 며칠씩 머물 수 있는지 알아본 후 놀러갈 섬을 결정하려 합니다.
지도를 나타내는 문자열 배열 maps가 매개변수로 주어질 때, 각 섬에서 최대 며칠씩 머무를 수 있는지 배열에 오름차순으로 담아 return 하는 solution 함수를 완성해주세요.
만약 지낼 수 있는 무인도가 없다면 -1을 배열에 담아 return 해주세요.
제한사항
3 ≤ maps의 길이 ≤ 100
3 ≤ maps[i]의 길이 ≤ 100
maps[i]는 'X' 또는 1 과 9 사이의 자연수로 이루어진 문자열입니다.
지도는 직사각형 형태입니다.
'''
from collections import deque
maps = list(map(str, input().split()))


def solution(maps):
    answer = []
    maps = [list(i) for i in maps]

    def bfs(y, x):
        q = deque()
        q.append((y, x))

        while q:
            y, x = q.popleft()
            if maps[y][x] == "X":
                continue
            day.append(int(maps[y][x]))
            maps[y][x] = "X"

            if y + 1 < len(maps) and maps[y + 1][x] != "X":
                q.append((y + 1, x))
            if y - 1 >= 0 and maps[y - 1][x] != "X":
                q.append((y - 1, x))
            if x + 1 < len(maps[0]) and maps[y][x + 1] != "X":
                q.append((y, x + 1))
            if x - 1 >= 0 and maps[y][x - 1] != "X":
                q.append((y, x - 1))
        return sum(day)

    for i in range(len(maps)):
        for j in range(len(maps[0])):
            if maps[i][j] != "X":
                day = []
                answer.append(bfs(i, j))

    if not answer:
        return [-1]
    return sorted(answer)


print(solution(maps))
