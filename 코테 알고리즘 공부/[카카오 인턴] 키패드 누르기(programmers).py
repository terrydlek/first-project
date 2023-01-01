'''
스마트폰 전화 키패드의 각 칸에 다음과 같이 숫자들이 적혀 있습니다.
이 전화 키패드에서 왼손과 오른손의 엄지손가락만을 이용해서 숫자만을 입력하려고 합니다.
맨 처음 왼손 엄지손가락은 * 키패드에 오른손 엄지손가락은 # 키패드 위치에서 시작하며, 엄지손가락을 사용하는 규칙은 다음과 같습니다.
엄지손가락은 상하좌우 4가지 방향으로만 이동할 수 있으며 키패드 이동 한 칸은 거리로 1에 해당합니다.
왼쪽 열의 3개의 숫자 1, 4, 7을 입력할 때는 왼손 엄지손가락을 사용합니다.
오른쪽 열의 3개의 숫자 3, 6, 9를 입력할 때는 오른손 엄지손가락을 사용합니다.
가운데 열의 4개의 숫자 2, 5, 8, 0을 입력할 때는 두 엄지손가락의 현재 키패드의 위치에서 더 가까운 엄지손가락을 사용합니다.
4-1. 만약 두 엄지손가락의 거리가 같다면, 오른손잡이는 오른손 엄지손가락, 왼손잡이는 왼손 엄지손가락을 사용합니다.
순서대로 누를 번호가 담긴 배열 numbers, 왼손잡이인지 오른손잡이인 지를 나타내는 문자열 hand가 매개변수로 주어질 때,
각 번호를 누른 엄지손가락이 왼손인 지 오른손인 지를 나타내는 연속된 문자열 형태로 return 하도록 solution 함수를 완성해주세요.
[제한사항]
numbers 배열의 크기는 1 이상 1,000 이하입니다.
numbers 배열 원소의 값은 0 이상 9 이하인 정수입니다.
hand는 "left" 또는 "right" 입니다.
"left"는 왼손잡이, "right"는 오른손잡이를 의미합니다.
왼손 엄지손가락을 사용한 경우는 L, 오른손 엄지손가락을 사용한 경우는 R을 순서대로 이어붙여 문자열 형태로 return 해주세요.
'''
numbers = list(map(int, input().split()))
hand = input()


def solution_1(numbers, hand):
    answer = ''
    r = 12
    l = 10
    li = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
          [10, 0, 12]]
    for i in numbers:
        if i == 1 or i == 4 or i == 7:
            answer += "L"
            l = i
        elif i == 3 or i == 6 or i == 9:
            answer += "R"
            r = i
        else:
            ri = ""
            le = ""
            key = ""
            for j in range(len(li)):
                for k in range(len(li[j])):
                    if li[j][k] == r:
                        ri = str(j) + str(k)
                    if li[j][k] == l:
                        le = str(j) + str(k)
                    if li[j][k] == i:
                        key = str(j) + str(k)
            if abs(int(ri[0]) - int(key[0])) + abs(int(ri[1]) - int(key[1])) < abs(int(le[0]) - int(key[0])) + \
                    abs(int(le[1]) - int(key[1])):
                answer += "R"
                r = i
            elif abs(int(ri[0]) - int(key[0])) + abs(int(ri[1]) - int(key[1])) > abs(int(le[0]) - int(key[0])) + \
                    abs(int(le[1]) - int(key[1])):
                answer += "L"
                l = i
            else:
                if hand == "right":
                    answer += "R"
                    r = i
                else:
                    answer += "L"
                    l = i
    return answer


def solution_2(numbers, hand):
    answer = ''
    location = [[3, 1], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    left, right = [3, 0], [3, 2]
    for i in numbers:
        if i % 3 == 1:
            answer += 'L'
            left = location[i]
        elif i % 3 == 0 and i != 0:
            answer += 'R'
            right = location[i]
        else:
            l = abs(location[i][0] - left[0]) + abs(location[i][1] - left[1])
            r = abs(location[i][0] - right[0]) + abs(location[i][1] - right[1])
            if l < r:
                answer += 'L'
                left = location[i]
            elif l > r:
                answer += 'R'
                right = location[i]
            else:
                answer += hand[0].upper()
                if hand == 'right':
                    right = location[i]
                else:
                    left = location[i]

    return answer


print(solution_1(numbers, hand))
print(solution_2(numbers, hand))
