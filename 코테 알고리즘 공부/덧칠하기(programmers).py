'''

'''
from collections import deque
n, m = map(int, input().split())
section = list(map(int, input().split()))


def solution(n, m, section):
    answer = 0
    section = deque(section)
    li = []
    if m == 1:
        return len(section)
    while section:
        li.append(deque.popleft(section))
        if len(li) == m or (li[-1] - li[0]) >= m:
            answer += 1
            section.insert(0, li[-1])
            li = []
        elif not section:
            answer += 1
    return answer


print(solution(n, m, section))
g