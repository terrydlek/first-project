'''
1부터 입력받은 숫자 n 사이에 있는 소수의 개수를 반환하는 함수, solution을 만들어 보세요.
소수는 1과 자기 자신으로만 나누어지는 수를 의미합니다.
(1은 소수가 아닙니다.)
'''
n = int(input())


def solution(n):
    answer = 0
    for i in range(2, n + 1):
        count = 0
        for j in range(2, int(i**(0.5)) + 1):
            if i % j == 0:
                count += 1
                break
        if count == 0:
            answer += 1
    return answer


print(solution(n))
