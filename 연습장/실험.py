from collections import deque
def solution(begin, target, words):
    answer = 0
    if target not in words:
        return 0
    q = deque()
    q.append((begin, 0))
    while q:
        bg, cnt = q.popleft()
        if bg == target:
            return cnt
        for i in words:
            count = 0
            # visited = ["o"] * len(words)
            for j in range(len(i)):
                if bg[j] != i[j]:
                    count += 1
            if count == 1 and bg == target:
                return cnt
            elif count == 1:
                q.append((i, cnt + 1))
    return answer


print(solution("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
