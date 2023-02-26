n = int(input())


def solution(n):
    answer = [[0 for j in range(i, 2 * i)] for i in range(1, n + 1)]
    행, 열 = -1, 0
    num = 1
    for i in range(n):
        for j in range(i, n):
            if i % 3 == 0:
                행 += 1
            elif i % 3 == 1:
                열 += 1
            else:
                행 -= 1
                열 -= 1
            answer[행][열] = num
            num += 1
            print(f'i: {i}, j: {j}, row: {행}, col: {열}, num: {num}')
            print(answer)
    return sum(answer, [])


print(solution(n))
