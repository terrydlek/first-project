'''
선분 3개가 평행하게 놓여 있습니다.
세 선분의 시작과 끝 좌표가 [[start, end], [start, end], [start, end]] 형태로 들어있는 2차원 배열 lines가 매개변수로 주어질 때,
두 개 이상의 선분이 겹치는 부분의 길이를 return 하도록 solution 함수를 완성해보세요.
lines가 [[0, 2], [-3, -1], [-2, 1]]일 때 그림으로 나타내면 다음과 같습니다.
선분이 두 개 이상 겹친 곳은 [-2, -1], [0, 1]로 길이 2만큼 겹쳐있습니다.
'''
lines = [list(map(int, input().split())) for _ in range(3)]


def solution(lines):
    answer = 0
    li = [0] * 201
    for i in lines:
        for j in range(i[0] + 100, i[1] + 101):
            li[j] += 1
    for k in range(1, len(li)):
        if li[k] > 1:
            if li[k - 1] > 1 and li.count(2) != 2:
                answer += 1
    return answer


print(solution(lines))
