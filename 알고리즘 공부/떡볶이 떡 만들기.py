'''n, m = map(int, input().split())
rice = list(map(int, input().split()))
rice.sort(reverse=True)

length = 0

for i in range(1,rice[0]):
  result = 0
  cut = rice[0] - i
  for j in rice:
    if j <= cut:
      pass
    else:
      result += (j % cut)
  if result >= m:
    length = cut
    break
print(length)
'''

# 다른 코드
n, m = list(map(int, input().split(' ')))
array = list(map(int, input().split()))
start = 0
end = max(array)

result = 0
while(start <= end):
  total = 0
  mid = (start + end) // 2
  for x in array:
    if x > mid:
      total += x - mid
  if total < m:
    end = mid - 1
  else:
    result = mid
    start = mid + 1
print(result)
