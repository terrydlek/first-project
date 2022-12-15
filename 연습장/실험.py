s = input()


def solution_2(s):
    answer = 0
    count = [0, 0]

    for i in s:
        if count[0] == count[1]:
            answer += 1
            alpha = i
        count[0 if i == alpha else 1] += 1
        print("answer: ", answer)
        print("alpha: ", alpha)
        print("count: ", count)
        print("i: ", i)
        print("--------------")
    return answer


print(solution_2(s))
