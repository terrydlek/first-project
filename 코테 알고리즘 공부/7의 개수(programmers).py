'''
머쓱이는 행운의 숫자 7을 가장 좋아합니다. 정수 배열 array가 매개변수로 주어질 때, 7이 총 몇 개 있는지 return 하도록 solution 함수를 완성해보세요.
'''
array = list(map(int, input().split()))


def solution_1(array):
    return sum([str(i).count("7") for i in array])


def solution_2(array):
    return ''.join(map(str, array)).count("7")


print(solution_1(array))
print(solution_2(array))
