x = int(input())
d = [0] * 30000

while x > 1:
  if x % 5 != 0 and x % 3 != 0:
    d[x - 1] = d[x] + 1
    x -= 1
  if x % 5 == 0:
    d[x//5] = d[x] + 1
    x //= 5
  if x % 5 != 0:
    d[x//3] = d[x] + 1
    x //= 3

print(d[1])

# 다른 풀이
'''x = int(input())

d = [0] * 30001

for i in range(2, x + 1):
  d[i] = d[i-1] + 1
  if i % 2 == 0:
    d[i] = min(d[i], d[i // 2] + 1)
  if i % 3 == 0:
    d[i] = min(d[i], d[i // 3] + 1)
  if i % 5 == 0:
    d[i] = min(d[i], d[i // 5] + 1)

print(d[x])
'''