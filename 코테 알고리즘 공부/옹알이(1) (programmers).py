'''
머쓱이는 태어난 지 6개월 된 조카를 돌보고 있습니다.
조카는 아직 "aya", "ye", "woo", "ma" 네 가지 발음을 최대 한 번씩 사용해 조합한(이어 붙인) 발음밖에 하지 못합니다.
문자열 배열 babbling이 매개변수로 주어질 때, 머쓱이의 조카가 발음할 수 있는 단어의 개수를 return하도록 solution 함수를 완성해주세요.
'''
babbling = list(map(str, input().split()))


def solution(babbling):
    answer = 0
    for i in babbling:
        possibility = ["aya", "ye", "woo", "ma"]
        string = ""
        start = 0
        for j in range(len(i)):
            if i[start:j + 1] in possibility:
                possibility.remove(i[start:j + 1])
                string += i[start:j + 1]
                start = j + 1
            if string == i:
                answer += 1
    return answer


print(solution(babbling))
