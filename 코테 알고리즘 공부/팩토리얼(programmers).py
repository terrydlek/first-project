'''
i팩토리얼 (i!)은 1부터 i까지 정수의 곱을 의미합니다. 예를들어 5! = 5 * 4 * 3 * 2 * 1 = 120 입니다.
정수 n이 주어질 때 다음 조건을 만족하는 가장 큰 정수 i를 return 하도록 solution 함수를 완성해주세요.
i! ≤ n
'''
n = int(input())


def solution_1(n):
    for i in range(10, 0, -1):
        count = 1
        for j in range(i, 0, -1):
            count *= j
        if count <= n:
            return i


from math import factorial


def solution_2(n):
    k = 10
    while n < factorial(k):
        k -= 1
    return k


print(solution_1(n))
print(solution_2(n))
