'''
자연수 x를 y로 변환하려고 합니다. 사용할 수 있는 연산은 다음과 같습니다.
x에 n을 더합니다
x에 2를 곱합니다.
x에 3을 곱합니다.
자연수 x, y, n이 매개변수로 주어질 때, x를 y로 변환하기 위해 필요한 최소 연산 횟수를 return하도록 solution 함수를 완성해주세요.
이때 x를 y로 만들 수 없다면 -1을 return 해주세요.
제한사항
1 ≤ x ≤ y ≤ 1,000,000
1 ≤ n < y
'''
from collections import deque
x, y, n = map(int, input().split())


def solution(x, y, n):
    q = deque()
    q.append((x, y, 0))
    while q:
        tg, num, cnt = q.popleft()
        if num == tg:
            return cnt
        if num % 2 == 0:
            q.append((tg, num // 2, cnt + 1))
        if num % 3 == 0:
            q.append((tg, num // 3, cnt + 1))
        if num - n >= 1:
            q.append((tg, num - n, cnt + 1))
    return -1


print(solution(x, y, n))
