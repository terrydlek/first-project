'''
2차원 좌표 평면에 변이 축과 평행한 직사각형이 있습니다.
직사각형 네 꼭짓점의 좌표 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]가 담겨있는 배열 dots가 매개변수로 주어질 때,
직사각형의 넓이를 return 하도록 solution 함수를 완성해보세요.
'''
dots = [list(map(int, input().split())) for i in range(4)]


def solution_1(dots):
    answer = 0
    for i in range(1, 4):
        if dots[0][0] != dots[i][0] and dots[0][1] != dots[i][1]:
            answer = abs(dots[0][0] - dots[i][0]) * abs(dots[0][1] - dots[i][1])
    return answer


def solution_2(dots):
    return (max(dots)[0] - min(dots)[0])*(max(dots)[1] - min(dots)[1])


print(solution_1(dots))
print(solution_2(dots))
