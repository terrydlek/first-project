import torch
'''w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
hypothesis = x_train*w + b
# H(x) = x 가 정확한 모델
# W = 1이 가장 좋은 숫자
# cost function <== 모델의 예측값이 실제의 데이터와 얼마나 다른지를 나타내는 함수
# 잘 학습된 모델일수록 낮은 cost를 가짐
cost = torch.mean((hypothesis - y_train) ** 2)
# cost function을 최소화해야 함
gradient = 2 * torch.mean((w * x_train - y_train) * x_train)
lr = 0.1
w -= lr * gradient
'''

'''# Full Code
# 데이터
# Epoch: 데이터로 학습한 횟수
# 학습하면서 점점 1에 수렴하는 W, 줄어드는 cost
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
w = torch.zeros(1)
# Learning rate 설정
lr = 0.1

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = x_train * w
    # cost gradient 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((w*x_train - y_train) * x_train)
    print("Epoch {:4d}/{} w: {:.3f}, Cost: {:6f}".format(epoch, nb_epochs, w.item(), cost.item()))
    # cost gradient로 H(x) 개선
    w -= lr * gradient'''

# torch.optim 으로도 gradient descent를 할 수 있음!
# 시작할 때 Optimizer 정의, optimizer.zero_grad()로 gradient를 0으로 초기화
# cost.backward()로 gradient 계산, optimizer.step()으로 gradient descent
# optimizer 설정
import torch.optim as optim
# optimizer = optim.SGD([w], lr=0.15)

'''# cost로 H(x) 개선
optimizer.zero_grad()
cost.backward()
optimizer.step()'''

# Full Code with torch.optim
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
w = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([w], lr=0.15)
nb_epochs = 10
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = x_train * w
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    print("Epoch {:4d}/{} w: {:.3f}, Cost: {:6f}".format(epoch, nb_epochs, w.item(), cost.item()))
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

# 지금까지 하나의 정보로부터 추측하는 모델을 만들었음
# ex 1) 수업 참여도 => 수업 점수, ex 2) 총 수면 시간 => 집중력
# but 대부분의 추측은 많은 정보를 추합해 이뤄짐
# ex 1) 쪽지 시험 성적ㅇ들 => 중간고사 성적, ex 2) 암의 위치, 넓이, 모양 => 치료 성공률
# 여러 개의 정보로부터 결론을 추측하는 모델은 어떻게 만드냐?
