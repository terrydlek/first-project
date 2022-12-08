'''
정수 배열 array와 정수 n이 매개변수로 주어질 때, array에 들어있는 정수 중 n과 가장 가까운 수를 return 하도록 solution 함수를 완성해주세요.
'''
array = list(map(int, input().split()))
n = int(input())


def solution(array, n):
    array.append(n)
    array.sort()
    a = array.index(n)
    if a == len(array) - 1:
        return array[a - 1]
    else:
        if abs(n - array[a - 1]) > abs(n - array[a + 1]):
            return array[a + 1]
        else:
            return array[a - 1]


print(solution(array, n))
