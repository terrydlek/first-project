'''
영어에선 a, e, i, o, u 다섯 가지 알파벳을 모음으로 분류합니다.
문자열 my_string이 매개변수로 주어질 때 모음을 제거한 문자열을 return하도록 solution 함수를 완성해주세요.
'''
my_string = input()


def solution_1(my_string):
    for i in my_string:
        if i == "a" or i == "e" or i == "i" or i == "o" or i == "u":
            my_string = my_string.replace(i, "")
    return my_string


def solution_2(my_string):
    return "".join([i for i in my_string if not(i in "aeiou")])


def solution_3(my_string):
    vowels = ['a', 'e', 'i', 'o', 'u']
    for vowel in vowels:
        my_string = my_string.replace(vowel, '')
    return my_string


print(solution_1(my_string))
print(solution_2(my_string))
print(solution_3(my_string))
