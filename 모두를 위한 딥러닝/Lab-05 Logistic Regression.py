import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# For reproducibility
torch.manual_seed(1)
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

print("e^1 equals: ", torch.exp(torch.FloatTensor([1])))

w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(w) + b)))
print(hypothesis)
print(hypothesis.shape)

print('1/(1 + e^{-1}) equals: ', torch.sigmoid(torch.FloatTensor([1])))
hypothesis = torch.sigmoid(x_train.matmul(w) + b)
print(hypothesis)
print(hypothesis.shape)

# Cost Function
# for one element, the loss can be computed as follows
print(-(y_train[0] * torch.log(hypothesis[0]) + (1 - y_train[0]) * torch.log(1 - hypothesis[0])))
losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
print(losses)

cost = losses.mean()
print(cost)

F.binary_cross_entropy(hypothesis, y_train)

# Whole Training Procedure
# 모델 초기화
w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([w, b], lr=1)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(w) + b)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print("epoch {:4d}/{} cost: {:.6f}".format(epoch, nb_epochs, cost.item()))

# evaluation
hypothesis = torch.sigmoid(x_train.matmul(w) + b)
print(hypothesis[:5])

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction[:5])

print(prediction[:5])
print(y_train[:5])

correct_prediction = prediction.float() == y_train
print(correct_prediction[:5])


# High-level Implementation with nn.Module
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = BinaryClassifier()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)
nb_epochs = 100
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = model(x_train)
    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print("Epoch {:4d}/{} cost: {:.6f} accuracy {:2.2f}%".format(epoch, nb_epochs, cost.item(), accuracy * 100,))