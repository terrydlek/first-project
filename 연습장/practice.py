def solution(players, callings):
    answer = []
    for i in callings:
        players[players.index(i)], players[players.index(i) - 1] = players[players.index(i) - 1], players[players.index(i)]
        print(players)
    return players

players = ["mumu", "soe", "poe", "kai", "mine"]
players[1], players[0] = players[0], players[1]
print(players)

# print(solution(["mumu", "soe", "poe", "kai", "mine"], ["kai", "kai", "mine", "mine"]))