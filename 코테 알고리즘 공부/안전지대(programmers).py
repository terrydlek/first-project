'''
다음 그림과 같이 지뢰가 있는 지역과 지뢰에 인접한 위, 아래, 좌, 우 대각선 칸을 모두 위험지역으로 분류합니다.
지뢰는 2차원 배열 board에 1로 표시되어 있고 board에는 지뢰가 매설 된 지역 1과, 지뢰가 없는 지역 0만 존재합니다.
지뢰가 매설된 지역의 지도 board가 매개변수로 주어질 때, 안전한 지역의 칸 수를 return하도록 solution 함수를 완성해주세요.
'''
n = int(input())
board = [list(map(int, input().split())) for _ in range(n)]


def solution_1(board):
    answer = 0
    li = [[0] * len(board) for _ in range(len(board))]
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 1:
                li[i][j] = 2
                if i - 1 >= 0:
                    li[i - 1][j] = 2
                    if j + 1 <= len(board) - 1:
                        li[i - 1][j + 1] = 2
                if i + 1 <= len(board) - 1:
                    li[i + 1][j] = 2
                    if j - 1 >= 0:
                        li[i + 1][j - 1] = 2
                if j + 1 <= len(board) - 1:
                    li[i][j + 1] = 2
                    if i + 1 <= len(board) - 1:
                        li[i + 1][j + 1] = 2
                if j - 1 >= 0:
                    li[i][j - 1] = 2
                    if i - 1 >= 0:
                        li[i - 1][j - 1] = 2
    for k in range(len(li)):
        answer += li[k].count(0)
    return answer


def solution_2(board):
    answer = 0

    for col in range(len(board)):
        for row in range(len(board[col])):
            if board[row][col] == 1:
                for j in range(max(col-1, 0), min(col+2, len(board))):
                    for i in range(max(row-1, 0), min(row+2, len(board))):
                        if board[i][j] == 1:
                            continue
                        board[i][j] = -1
    for i in board:
        answer += i.count(0)

    return answer


print(solution_1(board))
print(solution_2(board))
