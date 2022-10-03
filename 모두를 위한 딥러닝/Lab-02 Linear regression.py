import torch
import torch.optim as optim
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# weight와 bias 0 으로 초기화, 항상 출력 0을 예측
# requires_grad = True, 학습할 것이라고 명시
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = x_train*w + b

# Compute loss
cost = torch.mean((hypothesis - y_train) ** 2)
# torch.optim 라이브러리 사용, [w, b]는 학습할 tensor들, lr = 0.01은 learning rate
optimizer = optim.SGD([w, b], lr=0.01)
# zero_grad()로 gradient 초기화
optimizer.zero_grad()
# backward()로 gradient 계산
cost.backward()
# step()으로 개선
optimizer.step()

# Full training code
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 1.데이터 정의 2.hypothesis 초기화 3.optimizer 정의
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 반복. 1. hypothesis 예측 2. cost 계산 3. optimizer로 학습
optimizer = optim.SGD([w, b], lr=0.01)
nb_epochs = 1000
for epoch in range(1, nb_epochs + 1):
  hypothesis = x_train*w + b
  cost = torch.mean((hypothesis - y_train) ** 2)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()
