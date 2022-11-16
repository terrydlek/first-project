import heapq
s = "ababcdcdababcdcd"
for i in range(1, len(s)//2 + 1):
    li = []
    start = 0
    end = start + i
    while end < len(s) + i:
        li.append(s[start:end])
        start = end
        end = end + i
    start = 0
    end = start + i
    count = 1
    for j in range(len(li) - 1):
        if li[j] == li[j + 1]:
            count += 1
            li[j] = False
            if j == (len(li) - 2) and li[len(li) - 2] == False:
                li[j + 1] = str(count) + li[j + 1]
        else:
            if count == 1:
                pass
            else:
                li[j] = str(count) + li[j]
            count = 1
    print(li)
