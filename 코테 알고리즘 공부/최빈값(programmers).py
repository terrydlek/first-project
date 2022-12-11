'''
최빈값은 주어진 값 중에서 가장 자주 나오는 값을 의미합니다.
정수 배열 array가 매개변수로 주어질 때, 최빈값을 return 하도록 solution 함수를 완성해보세요. 최빈값이 여러 개면 -1을 return 합니다.
'''
array = list(map(int, input().split()))


def solution(array):
    s = [0] * 1001
    for i in array:
        s[i] = array.count(i)
    return -1 if s.count(max(s)) > 1 else s.index(max(s))


print(solution(array))
