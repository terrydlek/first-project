'''머쓱이네 피자가게는 피자를 일곱 조각으로 잘라 줍니다.
피자를 나눠먹을 사람의 수 n이 주어질 때, 모든 사람이 피자를 한 조각 이상 먹기 위해 필요한 피자의 수를 return 하는 solution 함수를 완성해보세요.'''
n = int(input())


def solution(n):
    answer = 0
    for i in range(1, n + 1):
        pizza = 7 * i
        if pizza // n >= 1:
            answer = i
            break
    return answer


print(solution(n))

# 다른 풀이
'''
def solution(n):
    return (n - 1) // 7 + 1
'''