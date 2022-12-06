'''
문자열 my_string과 정수 num1, num2가 매개변수로 주어질 때,
my_string에서 인덱스 num1과 인덱스 num2에 해당하는 문자를 바꾼 문자열을 return 하도록 solution 함수를 완성해보세요.
'''
my_string = input()
num1, num2 = map(int, input().split())


def solution(my_string, num1, num2):
    pr = list(my_string)
    pr[num1], pr[num2] = pr[num2], pr[num1]
    return "".join(pr)


print(solution(my_string, num1, num2))
