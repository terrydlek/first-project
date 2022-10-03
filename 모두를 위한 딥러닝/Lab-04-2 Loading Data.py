# Data in the Real World: Problem
# 엄청난 양의 데이터를 한번에 학습시킬 수 없다!
# 1. 너무 느림 2. 하드웨어적으로 불가능함
# 일부분의 데이터로만 학습하면 어떨까? ==> Minibatch Gradient Descent
# 업데이트를 좀 더 빠르게 할 수 있다.
# 전체 데이터를 쓰지 않아서 잘못된 방향으로 업데이트를 할 수도 있다.

# PyTorch Dataset
import torch
from torch.utils.data import Dataset
# torch.utils.data.Dataset 상속
# __len__() ==> 이 데이터셋의 총 데이터 수
# __getitem__() ==> 어떠한 인덱스 idx를 받았을 때, 그에 상응하는 입출력 데이터 반환


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

        def __len__(self):
            return len(self.x_data)

        def __getitem__(self, idx):
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y


dataset = CustomDataset()

from torch.utils.data import DataLoader
# torch.utils.data.DataLoader 사용
# batch_size = 2, 각 minibatch의 크기, 통상적으로 2의 제곱수로 설정함
# shuffle=True ==> Epoch마다 데이터셋을 섞어서, 데이터가 학습되는 순서를 바꾼다
# enumerate(dataloader) ==> minibatch 인덱스와 데이터를 받음
# len(dataloader) ==> 한 epoch당 minibatch 개수
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

import torch
import torch.nn.functional as F

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)
        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))
