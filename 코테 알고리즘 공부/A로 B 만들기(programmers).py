'''
문자열 before와 after가 매개변수로 주어질 때, before의 순서를 바꾸어 after를 만들 수 있으면 1을, 만들 수 없으면 0을 return 하도록 solution 함수를 완성해보세요.
'''
before = input()
after = input()


def solution_1(before, after):
    answer = 0
    for i in before:
        if i in after and after.count(i) == before.count(i):
            answer = 1
        else:
            answer = 0
            break
    return answer


def solution_2(before, after):
    return 1 if sorted(before) == sorted(after) else 0


print(solution_1(before, after))
print(solution_2(before, after))
