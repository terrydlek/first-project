'''
트럭 여러 대가 강을 가로지르는 일차선 다리를 정해진 순으로 건너려 합니다.
모든 트럭이 다리를 건너려면 최소 몇 초가 걸리는지 알아내야 합니다.
다리에는 트럭이 최대 bridge_length대 올라갈 수 있으며, 다리는 weight 이하까지의 무게를 견딜 수 있습니다.
단, 다리에 완전히 오르지 않은 트럭의 무게는 무시합니다.
예를 들어, 트럭 2대가 올라갈 수 있고 무게를 10kg까지 견디는 다리가 있습니다.
무게가 [7, 4, 5, 6]kg인 트럭이 순서대로 최단 시간 안에 다리를 건너려면 다음과 같이 건너야 합니다.
경과 시간	다리를 지난 트럭	다리를 건너는 트럭	대기 트럭
0	[]	[]	[7,4,5,6]
1~2	[]	[7]	[4,5,6]
3	[7]	[4]	[5,6]
4	[7]	[4,5]	[6]
5	[7,4]	[5]	[6]
6~7	[7,4,5]	[6]	[]
8	[7,4,5,6]	[]	[]
따라서, 모든 트럭이 다리를 지나려면 최소 8초가 걸립니다.
solution 함수의 매개변수로 다리에 올라갈 수 있는 트럭 수 bridge_length,
다리가 견딜 수 있는 무게 weight, 트럭 별 무게 truck_weights가 주어집니다.
이때 모든 트럭이 다리를 건너려면 최소 몇 초가 걸리는지 return 하도록 solution 함수를 완성하세요.
제한 조건
bridge_length는 1 이상 10,000 이하입니다.
weight는 1 이상 10,000 이하입니다.
truck_weights의 길이는 1 이상 10,000 이하입니다.
모든 트럭의 무게는 1 이상 weight 이하입니다.
'''
from collections import deque
bridge_length = int(input())
weight = int(input())
truck_weights = list(map(int, input().split()))


def solution_1(bridge_length, weight, truck_weights):
    answer = 0
    truck_queue = deque(truck_weights)
    bridge_queue = deque([])
    time_table = []
    #print(bridge_queue)
    #print(truck_queue)

    while True:
        weight_sum = 0

        if not truck_queue:
            if not bridge_queue:
                break
        answer += 1
        if bridge_queue:
            if bridge_queue[0][1] >= bridge_length:
                bridge_queue.popleft()

        if bridge_queue:
            for i in range(len(bridge_queue)):
                bridge_queue[i][1] += 1

        if bridge_queue:
            for i in range(len(bridge_queue)):
                weight_sum += bridge_queue[i][0]
        if truck_queue:
            if len(bridge_queue)+1 <= bridge_length and (weight_sum+truck_queue[0]) <= weight:
                #print(len(bridge_queue)+1,weight_sum+truck_queue[0])
                truck = truck_queue.popleft()
                bridge_queue.append([truck, 1])

        #print(bridge_queue)
        #print(truck_queue)
    return answer


def solution_2(bridge_length, weight, truck_weights):
    q = [0] * bridge_length
    sec = 0
    while q:
        sec += 1
        q.pop(0)
        if truck_weights:
            if sum(q)+truck_weights[0] <= weight:
                q.append(truck_weights.pop(0))
            else:
                q.append(0)
    return sec


print(solution_1(bridge_length, weight, truck_weights))
print(solution_2(bridge_length, weight, truck_weights))
