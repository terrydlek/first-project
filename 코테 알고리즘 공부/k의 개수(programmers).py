'''
1부터 13까지의 수에서, 1은 1, 10, 11, 12, 13 이렇게 총 6번 등장합니다.
정수 i, j, k가 매개변수로 주어질 때, i부터 j까지 k가 몇 번 등장하는지 return 하도록 solution 함수를 완성해주세요.
'''
i, j, k = map(int, input().split())


def solution_1(i, j, k):
    answer = 0
    for num in range(i, j + 1):
        answer += str(num).count(str(k))
    return answer


def solution_2(i, j, k):
    answer = sum([str(i).count(str(k)) for i in range(i, j + 1)])
    return answer


print(solution_1(i, j, k))
print(solution_2(i, j, k))
