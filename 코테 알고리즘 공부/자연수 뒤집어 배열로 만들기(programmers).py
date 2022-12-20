'''
자연수 n을 뒤집어 각 자리 숫자를 원소로 가지는 배열 형태로 리턴해주세요. 예를들어 n이 12345이면 [5,4,3,2,1]을 리턴합니다.
'''
n = int(input())


def solution_1(n):
    answer = []
    for i in str(n):
        answer.append(int(i))
    answer.reverse()
    return answer


def solution_2(n):
    return list(map(int, reversed(str(n))))


print(solution_1(n))
print(solution_2(n))
