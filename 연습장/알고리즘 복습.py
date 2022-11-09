n = int(input())
data = list(map(int, input().split()))
data.sort()

result = 1

for x in data:
    if result < x:
        break
    result += x

print(result)
