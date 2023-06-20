'''
rows x columns 크기인 행렬이 있습니다. 행렬에는 1부터 rows x columns까지의 숫자가 한 줄씩 순서대로 적혀있습니다.
이 행렬에서 직사각형 모양의 범위를 여러 번 선택해, 테두리 부분에 있는 숫자들을 시계방향으로 회전시키려 합니다.
각 회전은 (x1, y1, x2, y2)인 정수 4개로 표현하며, 그 의미는 다음과 같습니다.
x1 행 y1 열부터 x2 행 y2 열까지의 영역에 해당하는 직사각형에서 테두리에 있는 숫자들을 한 칸씩 시계방향으로 회전합니다.
행렬의 세로 길이(행 개수) rows, 가로 길이(열 개수) columns, 그리고 회전들의 목록 queries가 주어질 때,
각 회전들을 배열에 적용한 뒤, 그 회전에 의해 위치가 바뀐 숫자들 중 가장 작은 숫자들을 순서대로 배열에 담아 return 하도록 solution 함수를 완성해주세요.
제한사항
rows는 2 이상 100 이하인 자연수입니다.
columns는 2 이상 100 이하인 자연수입니다.
처음에 행렬에는 가로 방향으로 숫자가 1부터 하나씩 증가하면서 적혀있습니다.
즉, 아무 회전도 하지 않았을 때, i 행 j 열에 있는 숫자는 ((i-1) x columns + j)입니다.
queries의 행의 개수(회전의 개수)는 1 이상 10,000 이하입니다.
queries의 각 행은 4개의 정수 [x1, y1, x2, y2]입니다.
x1 행 y1 열부터 x2 행 y2 열까지 영역의 테두리를 시계방향으로 회전한다는 뜻입니다.
1 ≤ x1 < x2 ≤ rows, 1 ≤ y1 < y2 ≤ columns입니다.
모든 회전은 순서대로 이루어집니다.
예를 들어, 두 번째 회전에 대한 답은 첫 번째 회전을 실행한 다음, 그 상태에서 두 번째 회전을 실행했을 때 이동한 숫자 중 최솟값을 구하면 됩니다.
'''
rows, columns = map(int, input().split())
queries = [list(map(int, input().split())) for _ in range(rows)]


def solution(rows, columns, queries):
    answer = []
    li = []
    cnt = 1
    for _ in range(rows):
        lis = []
        for _ in range(columns):
            lis.append(cnt)
            cnt += 1
        li.append(lis)
    def change(li, sx, sy, ex, ey):
        re = []
        curx, cury = sx - 1, sy - 1
        pre = li[curx][cury]
        direction = "r"
        for _ in range((ex - sx + 1) * 2 + (ey - sy - 1) * 2):
            if direction == "r":
                if cury + 1 <= ey - 1:
                    cur = li[curx][cury + 1]
                    li[curx][cury + 1] = pre
                    pre = cur
                    cury += 1
                else:
                    direction = "d"
                    curx += 1
                    cur = li[curx][cury]
                    li[curx][cury] = pre
                    pre = cur
            elif direction == "d":
                if curx + 1 <= ex - 1:
                    cur = li[curx + 1][cury]
                    li[curx + 1][cury] = pre
                    pre = cur
                    curx += 1
                else:
                    direction = "l"
                    cury -= 1
                    cur = li[curx][cury]
                    li[curx][cury] = pre
                    pre = cur
            elif direction == "l":
                if cury - 1 >= sy - 1:
                    cur = li[curx][cury - 1]
                    li[curx][cury - 1] = pre
                    pre = cur
                    cury -= 1
                else:
                    direction = "u"
                    curx -= 1
                    cur = li[curx][cury]
                    li[curx][cury] = pre
                    pre = cur
            elif direction == "u":
                if curx - 1 >= sx - 1:
                    cur = li[curx - 1][cury]
                    li[curx - 1][cury] = pre
                    pre = cur
                    curx -= 1
            re.append(pre)
        return min(re)

    for i in queries:
        sx, sy, ex, ey = i
        answer.append(change(li, sx, sy, ex, ey))
    return answer


print(solution(rows, columns, queries))
