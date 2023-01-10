'''
정수 n, left, right가 주어집니다. 다음 과정을 거쳐서 1차원 배열을 만들고자 합니다.
n행 n열 크기의 비어있는 2차원 배열을 만듭니다.
i = 1, 2, 3, ..., n에 대해서, 다음 과정을 반복합니다.
1행 1열부터 i행 i열까지의 영역 내의 모든 빈 칸을 숫자 i로 채웁니다.
1행, 2행, ..., n행을 잘라내어 모두 이어붙인 새로운 1차원 배열을 만듭니다.
새로운 1차원 배열을 arr이라 할 때, arr[left], arr[left+1], ..., arr[right]만 남기고 나머지는 지웁니다.
정수 n, left, right가 매개변수로 주어집니다. 주어진 과정대로 만들어진 1차원 배열을 return 하도록 solution 함수를 완성해주세요.
'''
n, left, right = map(int, input().split())


def solution_1(n, left, right):
    answer = []
    for i in range(left, right + 1):
        answer.append(max(i // n, i % n) + 1)
    return answer


def solution_2(n, left, right):
    li = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        li[i][i] = i + 1
        for j in range(i):
            li[i][j] = i + 1
            li[j][i] = i + 1
    answer = []
    for i in li:
        for j in i:
            answer.append(j)
    print(answer)
    return answer[left:right + 1]


def solution_3(n, left, right):
    answer = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if j < i:
                answer.append(i)
            else:
                answer.append(j)
    return answer[left:right + 1]


print(solution_1(n, left, right))
print(solution_2(n, left, right))
print(solution_3(n, left, right))
