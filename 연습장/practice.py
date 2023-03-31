def solution(begin, target, words):
    answer = 0
    if target not in words:
        return 0

    def dfs(bg, tg, wd, answer):

        for i in words:
            count = 0
            for j in range(len(tg)):
                if bg[j] != i[j]:
                    count += 1

            if count == 1 and i == tg:
                return answer
            elif count == 1:
                print(bg, i, tg, answer, wd)
                dfs(i, tg, wd[:wd.index(i)] + wd[wd.index(i) + 1:], answer + 1)

    return dfs(begin, target, words, answer)


print(solution("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
