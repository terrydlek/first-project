from collections import deque


def solution(n=8, a=4, b=7):
    answer = 1
    li = deque([i for i in range(1, n + 1)])
    while len(li) > 2:
        for i in range(0, len(li), 2):
            if li[0] == a and li[1] == b:
                return answer
            elif li[0] == b and li[1] == a:
                return answer
            if li[0] == a or li[1] == a:
                li.append(a)
            elif li[0] == b or li[1] == b:
                li.append(b)
            else:
                li.append(li[0])
            deque.popleft(li)
            deque.popleft(li)
        answer += 1
        print(li, answer)
        print("==================")
    return answer


print(solution(8, 4, 7))
