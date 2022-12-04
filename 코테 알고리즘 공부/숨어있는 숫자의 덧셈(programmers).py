'''
문자열 my_string이 매개변수로 주어집니다. my_string안의 모든 자연수들의 합을 return하도록 solution 함수를 완성해주세요.
'''
my_string = input()


def solution_1(my_string):
    return sum(int(i) for i in my_string if i.isdigit())


def solution_2(my_string):
    answer = 0
    for i in my_string:
        if i == "1" or i == "2" or i == "3" or i == "4" or i == "5" or i == "6" or i == "7" or i == "8" or i == "9":
            answer += int(i)
    return answer


print(solution_1(my_string))
print(solution_2(my_string))
