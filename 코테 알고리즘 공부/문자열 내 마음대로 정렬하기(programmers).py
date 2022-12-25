'''
문자열로 구성된 리스트 strings와, 정수 n이 주어졌을 때, 각 문자열의 인덱스 n번째 글자를 기준으로 오름차순 정렬하려 합니다.
예를 들어 strings가 ["sun", "bed", "car"]이고 n이 1이면 각 단어의 인덱스 1의 문자 "u", "e", "a"로 strings를 정렬합니다.
'''
strings = list(map(str, input().split()))
n = int(input())


def solution(strings, n):
    answer = []
    li = []
    for i in strings:
        if ord(i[n]) not in li:
            li.append(ord(i[n]))
    li.sort()
    for j in li:
        count = 0
        lis = []
        for k in strings:
            if j == ord(k[n]):
                count += 1
                lis.append(k)
        if count == 1:
            answer.append(lis[0])
        else:
            lis.sort()
            for l in lis:
                answer.append(l)
    return answer


print(solution(strings, n))
