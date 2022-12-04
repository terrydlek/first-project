'''
정수 n이 매개변수로 주어질 때 n의 각 자리 숫자의 합을 return하도록 solution 함수를 완성해주세요
'''
n = int(input())


def solution_1(n):
    answer = 0
    a = str(n)
    for i in a:
        answer += int(i)
    return answer


def solution_2(n):
    return sum(list(map(int, str(n))))


print(solution_1(n))
print(solution_2(n))
