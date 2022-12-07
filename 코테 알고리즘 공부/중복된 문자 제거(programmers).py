'''
문자열 my_string이 매개변수로 주어집니다. my_string에서 중복된 문자를 제거하고 하나의 문자만 남긴 문자열을 return하도록 solution 함수를 완성해주세요.
'''
my_string = input()


def solution_1(my_string):
    s = list(my_string)
    q = []
    for i in range(len(s)):
        if s[i] in q:
            pass
        else:
            q.append(s[i])
    return "".join(q)


def solution_2(my_string):
    answer = ''
    for i in my_string:
        if i not in answer:
            answer += i
    return answer


print(solution_1(my_string))
print(solution_2(my_string))
