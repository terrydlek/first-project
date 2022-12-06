# 201811806 박제욱입니다. Convolution Encoder 과제 제출하겠습니다!
# Convolution Encoder
message = "1001101011"


def convolution_encoder(message):
    q = []
    for i in message:
        q.append(int(i))

    initial_code = [0, 0]
    answer = []
    while q:
        answer.append((q[0] + initial_code[0] + initial_code[1]) % 2)
        answer.append((q[0] + initial_code[1]) % 2)
        initial_code[1] = initial_code[0]
        initial_code[0] = q[0]
        q.pop(0)
    return answer


print(convolution_encoder(message))
