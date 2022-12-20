'''
행렬의 덧셈은 행과 열의 크기가 같은 두 행렬의 같은 행, 같은 열의 값을 서로 더한 결과가 됩니다.
2개의 행렬 arr1과 arr2를 입력받아, 행렬 덧셈의 결과를 반환하는 함수, solution을 완성해주세요.
'''
n = int(input())
arr1 = [list(map(int, input().split())) for _ in range(n)]
arr2 = [list(map(int, input().split())) for _ in range(n)]


def solution(arr1, arr2):
    answer = []
    for i in range(len(arr1)):
        li = []
        for j in range(len(arr1[0])):
            li.append(arr1[i][j] + arr2[i][j])
        answer.append(li)
    return answer


import numpy as np
def sumMatrix(A,B):
    A=np.array(A)
    B=np.array(B)
    answer=A+B
    return answer.tolist()


print(solution(arr1, arr2))
print(sumMatrix(arr1, arr2))
