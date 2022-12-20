'''
임의의 양의 정수 n에 대해, n이 어떤 양의 정수 x의 제곱인지 아닌지 판단하려 합니다.
n이 양의 정수 x의 제곱이라면 x+1의 제곱을 리턴하고, n이 양의 정수 x의 제곱이 아니라면 -1을 리턴하는 함수를 완성하세요.
'''
n = int(input())


def solution(n):
    return ((n**0.5) + 1)**2 if n**0.5 == int(n**0.5) else -1


print(solution(n))
