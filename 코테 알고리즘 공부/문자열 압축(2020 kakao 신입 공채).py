'''데이터 처리 전문가가 되고 싶은 '어피치'는 문자열을 압축하는 방법에 대해 공부를 하고 있습니다.
최근에 대량의 데이터 처리를 위한 간단한 비손실 압축 방법에 대해 공부를 하고 있는데, 문자열에서 같은 같이 연속해서 나타나는 것을
그 문자의 개수와 반복되는 값으로 표현하여 더 짧은 문자열로 줄여서 표현하는 알고리즘을 공부하고 있습니다.
간단한 예로 "aabbaccc"의 경우 "2a2b3c"(문자가 반복되지 않아 한 번만 나타난 경우 1은 생략함) 와 같이 표현할 수 있는데,
이러한 방식은 반복되는 문자가 적은 경우 압축률이 낮다는 단점이 있습니다. 예를 들면, "abcabcdede"와 같은 문자열은 전혀 압축되지 않습니다.
어피치는 이러한 단점을 해결하기 위해 문자열을 1개 이상의 단위로 잘라서 압축하여 더 짧은 문자열로 표현할 수 있는지 방법을 찾아보려 합니다.
예를 들어 "ababcdcdababcdcd"의 경우 문자를 1개 단위로 자르면 전혀 압축되지 않지만, 2개 단위로 잘라서 압축한다면 "2ababcdcd"로 표현할 수 있으며,
이 때가 가장 짧게 압축하여 표현할 수 있는 방법입니다.
다른 예로, "abcabcdede"와 같은 경우, 문자를 2개 단위로 잘라서 압축하면 "abcabc2de"가 되지만, 3개 단위로 자른다면 "2abcdede"가 되어 3개 단위가 가장 짧은 압축 방법이 됩니다.
이때 3개 단위로 자르고 마지막에는 남는 문자열은 그대로 붙여주면 됩니다.
압축할 문자열 s가 매개변수로 주어질 때, 위에 설명한 벙법으로 1개 이상 단위로 문자열을 잘라 압축하여 표현한 문자열 중 가장 짧은 것의 길이를 return하도록 solution 함수를 완성해주세요.'''
# https://school.programmers.co.kr/learn/courses/30/lessons/60057
# 내가 푼 코드 (28개중 하나가 런타임 에러 남)
s = input()


def solution(sequence):
    len_list = []
    for i in range(1, len(sequence)//2 + 1):
        data = []
        start = 0
        end = start + i
        while end < len(sequence) + i:
            data.append(sequence[start:end])
            start = end
            end = end + i
        count = 1
        for j in range(len(data) - 1):
            if data[j] == data[j + 1]:
                count += 1
                data[j] = False
                if j == (len(data) - 2) and data[len(data) - 2] == False:
                    data[j + 1] = str(count) + data[j + 1]
            else:
                if count == 1:
                    pass
                else:
                    data[j] = str(count) + data[j]
                count = 1
        string_count = ""
        for k in data:
            if k != False:
                string_count += k
        len_list.append(len(string_count))
    return min(len_list)


print(solution(s))

# 다른 코드
'''def solution(s):
    answer = len(s)
    # 1개 단위(step)부터 압축 단위를 늘려가며 확인
    for step in range(1, len(s)//2 + 1):
        compressed = ""
        # 앞에서부터 step만큼의 문자열 추출
        prev = s[0:step]
        count = 1
        # 단위(step) 크기만큼 증가시키며 이전 문자열과 비교
        for j in range(step, len(s), step):
            # 이전 상태와 동일하다면 압축 횟수(count) 증가
            if prev == s[j:j + step]:
                count += 1
            # 다른 문자열이 나왔다면(더 이상 압축하지 못하는 경우라면)
            else:
                compressed += str(count) + prev if count >= 2 else prev
                # 다시 상태 초기화
                prev = s[j:j + step]
                count = 1
        # 남아 있는 문자열에 대해서 처리
        compressed += str(count) + prev if count >= 2 else prev
        # 만들어지는 압축 문자열이 가장 짧은 것이 정답
        answer = min(answer, len(compressed))
    return answer
'''