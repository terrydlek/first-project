'''
문자열 s가 매개변수로 주어집니다. s에서 한 번만 등장하는 문자를 사전 순으로 정렬한 문자열을 return 하도록 solution 함수를 완성해보세요.
한 번만 등장하는 문자가 없을 경우 빈 문자열을 return 합니다.
'''
s = input()


def solution_1(s):
    answer = ''
    for i in s:
        if s.count(i) == 1:
            answer += i
    return "".join(sorted(answer))


def solution_2(s):
    answer = "".join(sorted([i for i in s if s.count(i) == 1]))
    return answer


print(solution_1(s))
print(solution_2(s))
