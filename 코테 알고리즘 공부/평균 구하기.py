'''정수를 담고 있는 배열 arr의 평균값을 return하는 함수, solution을 완성해보세요.'''
import numpy as np
arr = list(map(int, input().split()))


def solution(arr):
    return np.average(arr)


def average(arr):
    return sum(arr) / len(arr)


print(solution(arr))
print(average(arr))
