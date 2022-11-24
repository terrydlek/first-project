num = int(input())
b = 0
for i in range(2, num):
    if num % i == 0:
        b += 1
if b == 0:
    print("소수입니다.")
else:
    print("소수가 아닙니다.")
