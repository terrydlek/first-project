from collections import deque
import math


def solution(k, d):
    def cal(y, x, k, d):
        q = deque()
        visited = [["O"] * d for _ in range(d)]
        q.append((y, x))
        criteria = math.pow(d, 2)
        count = 0
        while q:
            chy, chx = q.popleft()
            count += 1

            if math.pow(chy + k, 2) + math.pow(chx, 2) <= criteria and visited[chy + k][chx] != "V":
                visited[chy + k][chx] = "V"
                q.append((chy + k, x))
            if math.pow(chy, 2) + math.pow(chx + k, 2) <= criteria and visited[chy][chx + k] != "V":
                visited[chy][chx + k] = "V"
                q.append((chy, chx + k))
            if math.pow(chy + k, 2) + math.pow(chx + k, 2) <= criteria and visited[chy + k][chx + k] != "V":
                visited[chy + k][chx + k] = "V"
                q.append((chy + k, chx + k))
        return count

    return cal(0, 0, k, d)

li = [1,2,3,4,5]
for i,j in enumerate(li):
    print(i, j)