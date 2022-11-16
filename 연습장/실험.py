s = input()


def solution(sequence):
    len_list = []
    for i in range(1, len(sequence)//2 + 1):
        data = []
        start = 0
        end = start + i
        while end < len(sequence) + i:
            data.append(sequence[start:end])
            start = end
            end = end + i
        count = 1
        for j in range(len(data) - 1):
            if data[j] == data[j + 1]:
                count += 1
                data[j] = False
                if j == (len(data) - 2) and data[len(data) - 2] == False:
                    data[j + 1] = str(count) + data[j + 1]
            else:
                if count == 1:
                    pass
                else:
                    data[j] = str(count) + data[j]
                count = 1
        string_count = ""
        for k in data:
            if k != False:
                string_count += k
        len_list.append(len(string_count))
    return min(len_list)


print(solution(s))
