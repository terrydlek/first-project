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
'''from collections import Counter
def solution(picks, minerals):
    answer = 0
    scope = Counter(minerals[:sum([i* 5  for i in picks])])
    avail = []
    for i in scope:
        avail.append(i)
    avail.sort()
    print(avail)
    print(scope)
    return answer'''

from collections import deque
def solution(picks, minerals):
    answer = 0
    ax = []
    for i in range(3):
        for _ in range(picks[i] * 5):
            if i == 0:
                ax.append('diamond')
            elif i == 1:
                ax.append('iron')
            elif i == 2:
                ax.append('stone')
    ax = deque(ax)
    print(ax)
    avail_mineral = deque(minerals[:len(ax)])
    print(avail_mineral)
    while len(ax) != 0 and len(avail_mineral) != 0:
        a, m = ax.popleft(), avail_mineral.popleft()
        if a == 'diamond':
            answer += 1
        elif a == 'iron' and m == 'diamond':
            answer += 5
        elif (a == 'iron' and m == 'iron') or (a == 'iron' and m == 'stone'):
            answer += 1
        elif a == 'stone' and m == 'diamond':
            answer += 25
        elif a == 'stone' and m == 'iron':
            answer += 5
        elif a == 'stone' and m == 'stone':
            answer += 1
        print(ax)
        print(avail_mineral)
        print(answer)
        print("============")
    return answer


print(solution([1, 3, 2], ["diamond", "diamond", "diamond", "iron", "iron", "diamond", "iron", "stone"]))
