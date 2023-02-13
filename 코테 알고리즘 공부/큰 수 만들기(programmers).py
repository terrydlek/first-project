'''
어떤 숫자에서 k개의 수를 제거했을 때 얻을 수 있는 가장 큰 숫자를 구하려 합니다.
예를 들어, 숫자 1924에서 수 두 개를 제거하면 [19, 12, 14, 92, 94, 24] 를 만들 수 있습니다. 이 중 가장 큰 숫자는 94 입니다.
문자열 형식으로 숫자 number와 제거할 수의 개수 k가 solution 함수의 매개변수로 주어집니다.
number에서 k 개의 수를 제거했을 때 만들 수 있는 수 중 가장 큰 숫자를 문자열 형태로 return 하도록 solution 함수를 완성하세요.
제한 조건
number는 2자리 이상, 1,000,000자리 이하인 숫자입니다.
k는 1 이상 number의 자릿수 미만인 자연수입니다.
'''
def solution(number, k):
    answer = ''
    count = 0
    number = list(number)
    length = len(number)
    while count < k:
        init = number[:k][0]

        print("number", number)
        print("answer", answer)
        print("init", init)
        print("count", count)
        print("===================")
        if k == 1:
            rang = number[:k + 1]
            for i in range(1, len(rang)):
                if int(rang[i]) > int(init):
                    if len(rang[:i]) + count <= k:
                        count += len(rang[:i])
                        for j in rang[:i]:
                            number.remove(j)
                        break
                    else:
                        for _ in range(k - count):
                            number.remove(min(rang))
                            count += 1
                            if count < k:
                                break
                elif i == k:
                    answer += number.pop(0)
            if len(answer) == length - k:
                return answer
        else:
            rang = number[:k]
            for i in range(k):
                if int(rang[i]) > int(init):
                    if len(rang[:i]) + count <= k:
                        count += len(rang[:i])
                        for j in rang[:i]:
                            number.remove(j)
                        break
                    else:
                        for _ in range(k - count):
                            number.remove(min(rang))
                            count += 1
                            if count < k:
                                break
                elif i == k - 1:
                    answer += number.pop(0)
            if len(answer) == length - k:
                return answer
    for i in number:
        answer += i
    print(number)
    return answer


print(solution("4321", 1))

