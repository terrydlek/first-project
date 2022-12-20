'''
정수 n을 입력받아 n의 약수를 모두 더한 값을 리턴하는 함수, solution을 완성해주세요.
'''
n = int(input())


def solution_1(n):
    answer = 0
    for i in range(1, n + 1):
        if n % i == 0:
            answer += i
    return answer


def solution_2(n):
    return sum([i for i in range(1, n + 1) if n % i == 0])


print(solution_1(n))
print(solution_2(n))
