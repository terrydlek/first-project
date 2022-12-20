'''
점 네 개의 좌표를 담은 이차원 배열  dots가 다음과 같이 매개변수로 주어집니다.
[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
주어진 네 개의 점을 두 개씩 이었을 때, 두 직선이 평행이 되는 경우가 있으면 1을 없으면 0을 return 하도록 solution 함수를 완성해보세요.
'''
dots = [list(map(int, input().split())) for _ in range(4)]


def solution(dots):
    answer = 0
    a = [[dots[0][0] - dots[1][0], dots[0][1] - dots[1][1]], [dots[2][0] - dots[3][0], dots[2][1] - dots[3][1]]]
    b = [[dots[0][0] - dots[2][0], dots[0][1] - dots[2][1]], [dots[1][0] - dots[3][0], dots[1][1] - dots[3][1]]]
    c = [[dots[0][0] - dots[3][0], dots[0][1] - dots[3][1]], [dots[1][0] - dots[2][0], dots[1][1] - dots[2][1]]]
    if a[0][1]/a[0][0] == a[1][1]/a[1][0]:
        answer = 1
    elif b[0][1]/b[0][0] == b[1][1]/b[1][0]:
        answer = 1
    elif c[0][1]/c[0][0] == c[1][1]/c[1][0]:
        answer = 1
    return answer


print(solution(dots))
