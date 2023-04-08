def solution(sequence, k):
    answer = []
    if k in sequence:
        return [sequence.index(k), sequence.index(k)]
    start, end = 0, 0
    sm = sequence[start]
    while True:
        if end != len(sequence) - 1:
            if sm == k:
                answer.append([start, end])
                sm -= sequence[start]
                start += 1
            elif sm < k:
                end += 1
                sm += sequence[end]
            elif sm > k:
                sm -= sequence[start]
                start += 1
        else:
            if sm == k:
                answer.append([start, end])
                sm -= sequence[start]
                start += 1
            else:
                sm -= sequence[start]
                start += 1
            if start == len(sequence) - 1:
                break
        print(answer, start, end, sm)
    return answer


print(solution([1, 1, 1, 2, 100], 102))
