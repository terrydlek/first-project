# 이진수 a, b
a = "11"
b = "1"
c = "22111"
# 이진수를 십진수로
A = int(a, 2)
B = int(b, 2)
# 3진법을 10진법으로
C = int(c, 3)
# int('숫자로 이루어진 문자열', 해당 진법) -> 10진법의 수로 변환
print(A)
print(B)
print(C)
print("==========================")
result = A + B
print(result)
# 십진수를 이진수로
result_ = format(result, "b")
print(result_)
