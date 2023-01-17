'''
게임 캐릭터를 4가지 명령어를 통해 움직이려 합니다. 명령어는 다음과 같습니다.
U: 위쪽으로 한 칸 가기
D: 아래쪽으로 한 칸 가기
R: 오른쪽으로 한 칸 가기
L: 왼쪽으로 한 칸 가기
캐릭터는 좌표평면의 (0, 0) 위치에서 시작합니다.
좌표평면의 경계는 왼쪽 위(-5, 5), 왼쪽 아래(-5, -5), 오른쪽 위(5, 5), 오른쪽 아래(5, -5)로 이루어져 있습니다.
예를 들어, "ULURRDLLU"로 명령했다면
1번 명령어부터 7번 명령어까지 다음과 같이 움직입니다.
8번 명령어부터 9번 명령어까지 다음과 같이 움직입니다.
이때, 우리는 게임 캐릭터가 지나간 길 중 캐릭터가 처음 걸어본 길의 길이를 구하려고 합니다.
예를 들어 위의 예시에서 게임 캐릭터가 움직인 길이는 9이지만, 캐릭터가 처음 걸어본 길의 길이는 7이 됩니다.
(8, 9번 명령어에서 움직인 길은 2, 3번 명령어에서 이미 거쳐 간 길입니다)
단, 좌표평면의 경계를 넘어가는 명령어는 무시합니다.
예를 들어, "LULLLLLLU"로 명령했다면
1번 명령어부터 6번 명령어대로 움직인 후, 7, 8번 명령어는 무시합니다. 다시 9번 명령어대로 움직입니다.
이때 캐릭터가 처음 걸어본 길의 길이는 7이 됩니다.
명령어가 매개변수 dirs로 주어질 때, 게임 캐릭터가 처음 걸어본 길의 길이를 구하여 return 하는 solution 함수를 완성해 주세요.
제한사항
dirs는 string형으로 주어지며, 'U', 'D', 'R', 'L' 이외에 문자는 주어지지 않습니다.
dirs의 길이는 500 이하의 자연수입니다.
'''
from collections import Counter
dirs = input()


def solution_1(dirs):
    answer = []
    lr = (1, 0)
    ud = (0, 1)
    first = (0, 0)
    for i in dirs:
        if i == "U":
            if first[1] + ud[1] <= 5 and first:
                back = first
                first = (first[0], first[1] + ud[1])
                if (back + first) not in answer and (first + back) not in answer:
                    answer.append(back + first)
        elif i == "D":
            if first[1] - ud[1] >= -5:
                back = first
                first = (first[0], first[1] - ud[1])
                if (back + first) not in answer and (first + back) not in answer:
                    answer.append(back + first)
        elif i == "L":
            if first[0] - lr[0] >= -5:
                back = first
                first = (first[0] - lr[0], first[1])
                if (back + first) not in answer and (first + back) not in answer:
                    answer.append(back + first)
        else:
            if first[0] + lr[0] <= 5:
                back = first
                first = (first[0] + lr[0], first[1])
                if (back + first) not in answer and (first + back) not in answer:
                    answer.append(back + first)
    co = Counter(answer)
    return len(co)


def solution_2(dirs):
    s = set()
    d = {'U': (0, 1), 'D': (0, -1), 'R': (1, 0), 'L': (-1, 0)}
    x, y = 0, 0
    for i in dirs:
        nx, ny = x + d[i][0], y + d[i][1]
        if -5 <= nx <= 5 and -5 <= ny <= 5:
            s.add((x, y, nx, ny))
            s.add((nx, ny, x, y))
            x, y = nx, ny
    return len(s)//2


print(solution_1(dirs))
print(solution_2(dirs))
