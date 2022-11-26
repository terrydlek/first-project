'''문자열 my_string과 문자 letter이 매개변수로 주어집니다. my_string에서 letter를 제거한 문자열을 return하도록 solution 함수를 완성해주세요.'''
my_string = input()
letter = input()

'''
def solution(my_string, letter):
    answer = ''
    for i in my_string:
        if i != letter:
            answer += i
    return answer
'''


def solution(my_string, letter):
    return my_string.replace(letter, '')


print(solution(my_string, letter))
