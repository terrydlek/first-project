'''from collections import deque


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


print(solution(["X591X", "X1X5X", "X231X", "1XXX1"]))
'''
v = [[0] * 8 for _ in range(4)]
print(v)
