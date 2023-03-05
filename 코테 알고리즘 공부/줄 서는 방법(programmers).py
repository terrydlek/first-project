'''
n명의 사람이 일렬로 줄을 서고 있습니다. n명의 사람들에게는 각각 1번부터 n번까지 번호가 매겨져 있습니다.
n명이 사람을 줄을 서는 방법은 여러가지 방법이 있습니다. 예를 들어서 3명의 사람이 있다면 다음과 같이 6개의 방법이 있습니다.
[1, 2, 3]
[1, 3, 2]
[2, 1, 3]
[2, 3, 1]
[3, 1, 2]
[3, 2, 1]
사람의 수 n과, 자연수 k가 주어질 때, 사람을 나열 하는 방법을 사전 순으로 나열 했을 때, k번째 방법을 return하는 solution 함수를 완성해주세요.
제한사항
n은 20이하의 자연수 입니다.
k는 n! 이하의 자연수 입니다.
'''
'''from itertools import permutations
n, k = map(int, input().split())


def solution_1(n, k):
    li = [i for i in range(1, n + 1)]
    answer = list(list(permutations(li, n))[k - 1])
    lis = [1, 2, 3, 4]
    print(list(permutations(lis, len(lis))))
    return answer


print(solution_1(n, k))'''
from math import factorial
n, k = map(int, input().split())


def solution(n, k):
    answer = []
    fac = factorial(n)
    li = [i for i in range(1, n + 1)]
    a = fac // n
    k -= 1
    while li:
        b = k // a
        answer.append(li[b % len(li)])
        li.pop(b % len(li))
        a = a // len(li)
        if len(li) == 1:
            answer.append(li[0])
            break
    return answer


print(solution(n, k))
