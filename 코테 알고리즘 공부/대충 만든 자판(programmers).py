'''

'''
keymap = list(map(str, input().split()))
targets = list(map(str, input().split()))


def solution(keymap, targets):
    answer = []
    dic = {i:100 for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        for j in keymap:
            if i in j:
                if j.index(i) < dic[i]:
                    dic[i] = j.index(i) + 1
    for i in targets:
        count = 0
        err = 0
        for j in i:
            if dic[j] == 100:
                err += 1
                answer.append(-1)
                break
            else:
                count += dic[j]
        if count != 0 and err == 0:
            answer.append(count)
    return answer


print(solution(keymap, targets))
