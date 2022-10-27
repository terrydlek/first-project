'''N, M = map(int, input().split()) #행(N)과 열(M) 입력받기
data = []
for i in range(N):
        data.append(list(map(int, input().split())))
min_data = []
for j in data:
    min_data.append(min(j))
print(max(min_data))
'''

#min() 함수를 이용
'''N, M = map(int, input().split())
result = 0
for i in range(N):
    data = map(int, input().split())
    min_data = min(data)
    result = max(result, min_data)

print(result)'''

#2중 반복문 사용
N, M = map(int, input().split())
result = 0

for i in range(N):
    data = list(map(int, input().split()))
    min_val = 10001
    for a in data:
        min_val = min(min_val, a)
    result = max(min_val, result)

print(result)
