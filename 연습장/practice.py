from itertools import permutations as pm


def solution(expression):
    answer = 0
    re = []
    rank = list(pm(["+", "-", "*"], 3))
    #print(list(rank))
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
    #print(string_li)

    for i in rank:
        li = string_li.copy()
        print(i)
        print(li)
        for j in i:
            while j in li:
                idx = li.index(j)
                cal = str(eval(li[idx - 1] + li[idx] + li[idx + 1]))
                li.insert(idx + 2, cal)
                del li[idx - 1]
                del li[idx - 1]
                del li[idx - 1]
                print(li)
        re.append(abs(int(li[0])))
    print(re)
    return max(re)


print(solution("100-200*300-500+20"))
