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
    print(li)
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
                    print(curx, cury)
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
            print(li, curx, cury, pre, direction)
        return min(re)

    for i in queries:
        sx, sy, ex, ey = i
        answer.append(change(li, sx, sy, ex, ey))
        print(answer)
        print("=======================")
    return answer


print(solution(3, 2, [[1,1,3,2]]))
