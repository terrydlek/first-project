'''
연속된 세 개의 정수를 더해 12가 되는 경우는 3, 4, 5입니다. 두 정수 num과 total이 주어집니다.
연속된 수 num개를 더한 값이 total이 될 때, 정수 배열을 오름차순으로 담아 return하도록 solution함수를 완성해보세요.
'''
num, total = map(int, input().split())


def solution_1(num, total):
    answer = []
    a = total // num
    b = total // num + 1
    add = 1
    minus = -1
    if num % 2 == 0:
        answer.append(a)
        answer.append(b)
        for _ in range((num - 1) // 2):
            answer.append(a + minus)
            answer.append(b + add)
            minus -= 1
            add += 1
    else:
        answer.append(a)
        for _ in range((num - 1) // 2):
            answer.append(a + minus)
            answer.append(a + add)
            minus -= 1
            add += 1
    return sorted(answer)


def solution_2(num, total):
    var = sum(range(num+1))
    diff = total - var
    start_num = diff//num
    answer = [i+1+start_num for i in range(num)]
    return answer


print(solution_1(num, total))
print(solution_2(num, total))
