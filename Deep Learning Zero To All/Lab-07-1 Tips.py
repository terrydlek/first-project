import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For reproducibility
torch.manual_seed(1)
x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])
x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)


def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        # H(x) 계산
        prediction = model(x_train)
        # cost 계산
        cost = F.cross_entropy(prediction, y_train)
        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, nb_epochs, cost.item()))


def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print("Accuracy: {} % Cost: {:.6f}".format(correct_count / len(y_test) * 100, cost.item()))


train(model, optimizer, x_train, y_train)
test(model, optimizer, x_test, y_test)
# learning rate이 너무 크면 diverge 하면서 cost가 점점 늘어남(overfitting)
model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=1e5)
train(model, optimizer, x_train, y_train)

# learning rate이 너무 작으면 cost가 거의 줄어들지 않는다
model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=1e-10)
train(model, optimizer, x_train, y_train)

# 적절한 숫자로 시작해 발산하면 작게, cost가 줄어들지 않으면 크게 조정하자
model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=1e-1)
train(model, optimizer, x_train, y_train)


# Data Preprocessing (데이터 전처리)
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
print(sigma)
norm_x_train = (x_train - mu) / sigma
print(norm_x_train)


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-1)


# x_train = (m, 3) prediction = (m, 1)
def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        # H(x) 계산
        prediction = model(x_train)
        # cost 계산
        cost = F.mse_loss(prediction, y_train) # 차이 계산
        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step() # gradient descent
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


train(model, optimizer, norm_x_train, y_train)
