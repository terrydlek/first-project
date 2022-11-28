# 이진수 a, b
a = "11"
b = "1"
# 이진수를 십진수로
A = int(a, 2)
B = int(b, 2)
print(A)
print(B)
result = A + B
print(result)
# 십진수를 이진수로
result_ = format(result, "b")
print(result_)
