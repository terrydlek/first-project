def solution(files):
    answer = []
    head = []
    # head, head가 아닌 것으로 분할
    for i in files:
        a = []
        string = ""
        for j in i:
            if j.isdigit():
                a.append(string)
                a.append(i[i.index(j):])
                head.append(a)
                break
            else:
                string += j
    print("분할: ", head)
    # head로 조건에 맞게 정렬
    for i in range(len(head) - 1):
        for j in range(i + 1, len(head)):
            a = head[i][0].lower()
            b = head[j][0].lower()
            if a > b:
                head[i], head[j] = head[j], head[i]
    print("head 정렬: ", head)
    # head가 아닌 부분으로 다시 정렬
    for i in range(len(head) - 1):
        for k in range(i + 1, len(head)):
            fir_string = ""
            sec_string = ""
            for j in head[i][1]:
                if j.isdigit() and len(fir_string) < 5:
                    fir_string += j
                else:
                    break
            for h in head[k][1]:
                if h.isdigit() and len(sec_string) < 5:
                    sec_string += h
                else:
                    break
            if int(fir_string) > int(sec_string) and head[i][0].lower() >= head[k][0].lower():
                head[i], head[k] = head[k], head[i]
    print("head가 아닌 부분 정렬: ", head)
    # head와 number도 같을 경우 원래 입력 순서 유지
    for i in range(len(head) - 1):
        for j in range(i + 1, len(head)):
            fir_string = ""
            sec_string = ""
            for k in head[i][1]:
                if k.isdigit() and len(fir_string) < 5:
                    fir_string += k
                else:
                    break
            for h in head[j][1]:
                if h.isdigit() and len(sec_string) < 5:
                    sec_string += h
            if head[i][0].lower() == head[j][0].lower() and int(fir_string) == int(sec_string):
                if files.index(head[i][0] + head[i][1]) > files.index(head[j][0] + head[j][1]):
                    head[i], head[j] = head[j], head[i]
    print("입력 순서 유지: ", head)
    for i in head:
        answer.append((i[0] + i[1]))
    return answer