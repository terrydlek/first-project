'''
2차원 행렬 arr1과 arr2를 입력받아, arr1에 arr2를 곱한 결과를 반환하는 함수, solution을 완성해주세요.
제한 조건
행렬 arr1, arr2의 행과 열의 길이는 2 이상 100 이하입니다.
행렬 arr1, arr2의 원소는 -10 이상 20 이하인 자연수입니다.
곱할 수 있는 배열만 주어집니다.
'''
import numpy as np
n, m = map(int, input().split())
arr1 = [list(map(int, input().split())) for _ in range(n)]
arr2 = [list(map(int, input().split())) for _ in range(m)]


def solution(arr1, arr2):
    answer = []
    a = np.dot(arr1, arr2)
    for i in a:
        b = []
        for j in range(len(i)):
            b.append(int(i[j]))
        answer.append(b)
    return answer


def productMatrix(A, B):
    answer = []
    for y1 in range(len(A)):
        a=[]
        for x2 in range(len(B[0])):
            n = 0
            for x1 in range(len(A[0])):
                n += A[y1][x1] * B[x1][x2]
            a.append(n)
        answer.append(a)
    return answer


print(solution(arr1, arr2))
print(productMatrix(arr1, arr2))
