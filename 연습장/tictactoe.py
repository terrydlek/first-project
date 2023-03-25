import random

def play_random(state):
    """
    현재 상태 state를 입력으로 받아 비어있는 임의의 좌표를 반환하는 함수
    """
    empty_cells = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0]
    if empty_cells:
        return random.choice(empty_cells)
    else:
        return None

def play_computer_turn(board, player_mark):
    """
    Tic-Tac-Toe 게임에서 computer의 차례를 처리하는 함수
    board: 게임판 상태
    player_mark: computer의 플레이어 마크
    """
    print("컴퓨터가 수를 놓습니다.")
    computer_move = play_random(board)
    board[computer_move[0]][computer_move[1]] = player_mark
    print(board)

