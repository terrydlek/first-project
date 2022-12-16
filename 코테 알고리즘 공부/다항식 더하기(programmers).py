'''
한 개 이상의 항의 합으로 이루어진 식을 다항식이라고 합니다. 다항식을 계산할 때는 동류항끼리 계산해 정리합니다.
덧셈으로 이루어진 다항식 polynomial이 매개변수로 주어질 때, 동류항끼리 더한 결괏값을 문자열로 return 하도록 solution 함수를 완성해보세요.
같은 식이라면 가장 짧은 수식을 return 합니다.
'''
polynomial = input()


def solution(polynomial):
    answer = [0, 0]
    for i in polynomial.split(" "):
        if "x" in i:
            if len(i) == 1:
                answer[0] += 1
            else:
                answer[0] += int(i[:i.index("x")])
        elif i.isdigit():
            answer[1] += int(i)
    if answer[1] == 0:
        return str(answer[0]) + "x" if answer[0] != 1 else "x"
    elif answer[0] == 0:
        return str(answer[1])
    else:
        return (str(answer[0]) + "x" if answer[0] != 1 else "x") + " " + "+" + " " + str(answer[1])


print(solution(polynomial))
