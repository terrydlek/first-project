'''
my_string은 "3 + 5"처럼 문자열로 된 수식입니다. 문자열 my_string이 매개변수로 주어질 때, 수식을 계산한 값을 return 하는 solution 함수를 완성해주세요.
'''
my_string = input()


def solution_1(my_string):
    s = ""
    cal = []
    num = []
    for i in range(len(my_string)):
        if my_string[i].isdigit():
            s += my_string[i]
            if i == len(my_string) - 1:
                num.append(int(s))
        elif my_string[i] == "+" or my_string[i] == "-":
            cal.append(my_string[i])
            num.append(int(s))
            s = ""
    answer = num[0]
    for j in range(len(cal)):
        if cal[j] == "+":
            answer += num[j + 1]
        else:
            answer -= num[j + 1]
    return answer


def solution_2(my_string):
    s = my_string.split()
    answer = int(s[0])
    for i in range(1, len(s), 2):
        if s[i] == "+":
            answer += int(s[i + 1])
        else:
            answer -= int(s[i + 1])
    return answer


def solution_3(my_string):
    return eval(my_string)


print(solution_1(my_string))
print(solution_2(my_string))
print(solution_3(my_string))
