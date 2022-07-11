''''#N(배열의 크기),M(숫자가 더해지는 횟수),K(연속해서 더해질 수 있는 횟수)를 공백으로 구분하여 입력받기
N, M, K = map(int, input("N(배열의 크기),M(숫자가 더해지는 횟수),K(연속해서 더해질 수 있는 횟수)를 공백으로 구분하여 입력받기: ").split())
data = list(map(int, input("데이터 입력받기").split()))

if K > M:
    print("다시 입력해주세요")
    N, M, K = map(int, input("N(배열의 크기),M(숫자가 더해지는 횟수),K(연속해서 더해질 수 있는 횟수)를 공백으로 구분하여 입력받기: ").split())
elif len(data) != N:
    print("다시 입력해주세요")
    data = list(map(int, input("데이터 입력받기: ").split()))
else:
    print("시작")

data.sort()
first = data[N-1]
second = data[N-2]
result = 0
while True:
    for i in range(K):
        if M == 0:
            break
        result += first
        M -= 1
    if M == 0:
        break
    result += second
    M -= 1
print(result)
'''
#더 깔끔하게 푸는 방법
N, M, K = map(int, input().split())
data = list(map(int, input().split()))
data.sort()
first = data[N-1]
second = data[N-2]
result = (first*K + second) * (M//(K+1)) + first * (M % (K+1))
print(result)
