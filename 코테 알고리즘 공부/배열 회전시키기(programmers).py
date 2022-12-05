'''
정수가 담긴 배열 numbers와 문자열 direction가 매개변수로 주어집니다.
배열 numbers의 원소를 direction방향으로 한 칸씩 회전시킨 배열을 return하도록 solution 함수를 완성해주세요.
'''
numbers = list(map(int, input().split()))
direction = input()


def solution_1(numbers, direction):
    answer = []
    if direction == "right":
        answer.append(numbers[-1])
        for j in range(len(numbers) - 1):
            answer.append(numbers[j])
    else:
        for j in range(1, len(numbers)):
            answer.append(numbers[j])
        answer.append(numbers[0])
    return answer


def solution_2(numbers, direction):
    return [numbers[-1]] + numbers[:-1] if direction == "right" else numbers[1:] + [numbers[0]]


print(solution_1(numbers, direction))
print(solution_2(numbers, direction))
