#while문 이용
def factorial(i):
    result = 1
    while i != 0:
        result *= i
        i -= 1
    return print(result)

#for문 이용
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return print(result)

#재귀적으로 구현
def factorial_recursive(k):
    if k <= 1:
        return 1
    return k * factorial_recursive(k-1)
