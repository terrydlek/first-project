'''
마인은 곡괭이로 광산에서 광석을 캐려고 합니다. 마인은 다이아몬드 곡괭이, 철 곡괭이, 돌 곡괭이를 각각 0개에서 5개까지 가지고 있으며,
곡괭이로 광물을 캘 때는 피로도가 소모됩니다. 각 곡괭이로 광물을 캘 때의 피로도는 아래 표와 같습니다.
예를 들어, 철 곡괭이는 다이아몬드를 캘 때 피로도 5가 소모되며, 철과 돌을 캘때는 피로도가 1씩 소모됩니다.
각 곡괭이는 종류에 상관없이 광물 5개를 캔 후에는 더 이상 사용할 수 없습니다.
마인은 다음과 같은 규칙을 지키면서 최소한의 피로도로 광물을 캐려고 합니다.
사용할 수 있는 곡괭이중 아무거나 하나를 선택해 광물을 캡니다.
한 번 사용하기 시작한 곡괭이는 사용할 수 없을 때까지 사용합니다.
광물은 주어진 순서대로만 캘 수 있습니다.
광산에 있는 모든 광물을 캐거나, 더 사용할 곡괭이가 없을 때까지 광물을 캡니다.
즉, 곡괭이를 하나 선택해서 광물 5개를 연속으로 캐고, 다음 곡괭이를 선택해서 광물 5개를 연속으로 캐는 과정을 반복하며,
더 사용할 곡괭이가 없거나 광산에 있는 모든 광물을 캘 때까지 과정을 반복하면 됩니다.
마인이 갖고 있는 곡괭이의 개수를 나타내는 정수 배열 picks와 광물들의 순서를 나타내는 문자열 배열 minerals가 매개변수로 주어질 때,
마인이 작업을 끝내기까지 필요한 최소한의 피로도를 return 하는 solution 함수를 완성해주세요.
제한사항
picks는 [dia, iron, stone]과 같은 구조로 이루어져 있습니다.
0 ≤ dia, iron, stone ≤ 5
dia는 다이아몬드 곡괭이의 수를 의미합니다.
iron은 철 곡괭이의 수를 의미합니다.
stone은 돌 곡괭이의 수를 의미합니다.
곡괭이는 최소 1개 이상 가지고 있습니다.
5 ≤ minerals의 길이 ≤ 50
minerals는 다음 3개의 문자열로 이루어져 있으며 각각의 의미는 다음과 같습니다.
diamond : 다이아몬드
iron : 철
stone : 돌
'''
picks = list(map(int, input().split()))
minerals = list(map(str, input().split()))


def solution(picks, minerals):
    answer = 0
    dct = {'diamond': 0, 'iron': 1, 'stone': 2}
    avail_min = minerals[:sum(picks) * 5]

    rng = [[0, 0, 0] for _ in range((len(avail_min) - 1) // 5 + 1)]
    for i in range(len(avail_min)):
        rng[i // 5][dct[avail_min[i]]] += 1

    rng.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    ax = ["diamond" for _ in range(picks[0])] + ["iron" for _ in range(picks[1])] + ["stone" for _ in range(picks[2])]

    for i in range(len(rng)):
        if ax[i] == "diamond":
            answer += sum(rng[i])
        elif ax[i] == "iron":
            answer += rng[i][0] * 5 + rng[i][1] + rng[i][2]
        else:
            answer += rng[i][0] * 25 + rng[i][1] * 5 + rng[i][2]
    return answer


print(solution(picks, minerals))
