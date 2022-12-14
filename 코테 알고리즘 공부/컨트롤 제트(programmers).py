'''
숫자와 "Z"가 공백으로 구분되어 담긴 문자열이 주어집니다.
문자열에 있는 숫자를 차례대로 더하려고 합니다. 이 때 "Z"가 나오면 바로 전에 더했던 숫자를 뺀다는 뜻입니다.
숫자와 "Z"로 이루어진 문자열 s가 주어질 때, 머쓱이가 구한 값을 return 하도록 solution 함수를 완성해보세요.
'''
s = input()


def solution_1(s):
    answer = 0
    a = ""
    b = []
    for i in range(len(s)):
        if s[i].isdigit() or s[i] == "-":
            a += s[i]
            if i == len(s) - 1:
                b.append(a)
        elif s[i] == "Z":
            b.append(a)
            b.append("-")
            a = ""
        else:
            b.append(a)
            a = ""
    for j in range(len(b)):
        if b[j].isdigit() or len(b[j]) >= 2:
            answer += int(b[j])
        elif b[j] == "-":
            answer -= int(b[j - 2])
    return answer


def solution_2(s):
    answer = 0
    stack = []
    for c in s.split():
        if c != 'Z':
            answer += int(c)
            stack.append(int(c))
        elif stack:
            answer -= stack.pop()
    return answer


print(solution_1(s))
print(solution_2(s))
