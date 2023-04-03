from collections import deque
def solution(triangle):
    answer = 0
    q = deque()
    q.append((0, 0, triangle[0][0]))
    li = []
    while q:
        y, x, sm = q.popleft()
        if y + 1 == len(triangle):
            li.append(sm)
            continue
        for i in triangle[y + 1]:
            q.append((y + 1, i, sm + i))
    print(li)
    return answer


print(solution([[7], [3, 8]]))
# print(solution([[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]]))