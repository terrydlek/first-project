'''
문자열 s를 숫자로 변환한 결과를 반환하는 함수, solution을 완성하세요.
'''
s = input()


def solution(s):
    answer = 0
    if s[0] == "+":
        answer = int(s[1:])
    elif s[0].isdigit():
        answer = int(s)
    elif s[0] == "-":
        answer = -1 * int(s[1:])
    return answer


print(solution(s))
