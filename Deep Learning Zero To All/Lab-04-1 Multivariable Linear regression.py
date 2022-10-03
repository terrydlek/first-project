# matmul()로 한번에 계산할 수 있음
# 더 간결하고, x의 길이가 바뀌어도 코드를 바꿀 필요가 없고, 속도도 더 빠르다!
# multiple linear regression도 simple linear regression과 cost를 구하는 공식이 동일
# cost = torch.mean((hypothesis - y_train)**2)
# Full Code with torch.optim(1)
# 데이터
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 모델 초기화
w = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([w, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = x_train.matmul(w) + b
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print("Epoch {:4d}/{} hypothesis: {} Cost: {:6f}".format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))

# nn.Module
# 모델 초기화
w = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# H(x) 계산
hypothesis = x_train.matmul(w) + b # or .mm or @


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

# hypothesis = model(x_train)


"""nn.Module을 상속해서 모델 생성
nn.Linear(3, 1) ==> 입력 차원: 3, 출력 차원: 1
Hypothesis 계산은 forward()에서 !
Gradient 계산은 Pytorch가 알아서 해준다 ==> backward()"""

# F.mse_loss
# cost 계산
'''cost = torch.mean((hypothesis - y_train) ** 2)
cost = F.mse_loss(prediction, y_train)'''
"""torch.nn.functional에서 제공하는 loss function 사용
쉽게 다른 loss와 교체 가능! (l1_loss, smooth_l1_loss 등...)"""

# nn.Module과 F.mse_loss가 적용된 코드
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 모델 초기화
# w = torch.zeros((3, 1), requires_grad=True) 필요없음
# b = torch.zeros(1, requires_grad=True) 필요없음
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD([w, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    prediction = model(x_train)
    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()))
# 지금까지는 적은 양의 데이터를 가지고 학습했음
# but 딥러닝은 많은 양의 데이터와 함께할 때 빛을 발함
# Pytorch에서는 많은 양의 데이터를 어떻게 다룰까?
