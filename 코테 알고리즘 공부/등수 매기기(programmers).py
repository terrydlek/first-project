'''
영어 점수와 수학 점수의 평균 점수를 기준으로 학생들의 등수를 매기려고 합니다.
영어 점수와 수학 점수를 담은 2차원 정수 배열 score가 주어질 때,
영어 점수와 수학 점수의 평균을 기준으로 매긴 등수를 담은 배열을 return하도록 solution 함수를 완성해주세요.
'''
n = int(input())
score = [list(map(int, input().split())) for _ in range(n)]


def solution_1(score):
    answer = []
    for i in score:
        answer.append(sum(i)/2)
    a = sorted(answer, reverse=True)
    result = [1]
    for j in range(1, len(a)):
        result.append(j + 1)
        if a[j] == a[j - 1]:
            result[j] = result[j - 1]
    b = [0] * len(a)
    for k in range(len(answer)):
        b[k] = result[a.index(answer[k])]
    return b


def solution_2(score):
    a = sorted([sum(i) for i in score], reverse=True)
    return [a.index(sum(i))+1 for i in score]


print(solution_1(score))
print(solution_2(score))
