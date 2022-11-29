# 피보나치 수열 재귀함수로 구현
def fibo_1(x):
  if x == 1 or x == 2:
    return 1
  return fibo_1(x - 1) + fibo_1(x - 2)


print(fibo_1(6))
# but 피보나치를 이렇게 작성하면 x가 커질수록 수행 시간이 기하급수적으로 늘어남
# fibo(100)을 계산하려면 1,000,000,000,000,000,000,000,000,000,000번 계산해야함

# 한 번 계산된 결과를 메모이제이션하기 위한 리스트 초기화
d = [0] * 100
# 피보나치 함수를 재귀함수로 구현(탑다운 다이나믹 프로그래밍)


def fibo_2(x):
  if x == 1 or x == 2:
    print("f(" + str(x) + ')', end=' ')
    return 1
# 이미 계산한 적 있는 문제라면 그대로 반환
  if d[x] != 0:
    return d[x]
# 아직 계산하지 않은 문제라면 점화식에 따라서 피보나치 결과 반환
  d[x] = fibo_2(x - 1) + fibo_2(x - 2)
  return d[x]

def fibo_3(x):
  num1 = 0
  num2 = 1
  num3 = 0
  if x == 0:
    num2 = 0
  else:
    for i in range(x - 1):
      num3 = num2
      num2 = num1 + num2
      num1 = num3
  return num2

print(fibo_2(99))
print(fibo_3(5))
array = [0] * 100
array[1] = 1
array[2] = 1
n = 99

for i in range(3, n+1):
  array[i] = array[i-1] + array[i-2]

print(array[n])
