from itertools import permutations as pm
expression = input()


def solution(expression):
    re = []
    rank = list(pm(["+", "-", "*"], 3))
    string = ""
    string_li = []
    for i in expression:
        if i.isdigit():
            string += i
        else:
            string_li.append(string)
            string_li.append(i)
            string = ""
    if string:
        string_li.append(string)

    for i in rank:
        li = string_li.copy()
        for j in i:
            while j in li:
                idx = li.index(j)
                cal = str(eval(li[idx - 1] + li[idx] + li[idx + 1]))
                li.insert(idx + 2, cal)
                del li[idx - 1]
                del li[idx - 1]
                del li[idx - 1]
        re.append(abs(int(li[0])))
    return max(re)


print(solution(expression))
