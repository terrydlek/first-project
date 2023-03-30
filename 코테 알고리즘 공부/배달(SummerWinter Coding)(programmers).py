from collections import deque


def solution(N, road, K):
    answer = 1
    a = [[] for _ in range(N + 1)]
    for i in road:
        x, y, d = i
        if x < y:
            a[x].append([x, y, d])
        else:
            a[y].append([y, x, d])

    def cal(end):
        q = deque()
        for i in a[1]:
            q.append(i)
        while q:
            st, ed, ds = q.popleft()
            if ed == end and ds <= K:
                return True
            if ds > K:
                continue
            if a[ed]:
                for j in a[ed]:
                    if j[2] + ds <= K:
                        print([j[0], j[1], j[2] + ds])
                        q.append([j[0], j[1], j[2] + ds])
        return False
    print(cal(3))
    for i in range(1, N + 1):
        if cal(i) == True:
            answer += 1
    return answer


print(solution(6, [[1,2,1],[1,3,2],[2,3,2],[3,4,3],[3,5,2],[3,5,3],[5,6,1]], 4))

