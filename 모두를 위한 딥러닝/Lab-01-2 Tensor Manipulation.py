import torch
import numpy as np
# View
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
              [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
print(ft.view([-1, 3])) # 앞에는 모르겠고 뒤의 차원은 3개의 element를 가져라
print(ft.view([-1, 3]).shape)
print(ft.view([-1, 1, 3])) # 바꿀 때 2*2*3이랑 일치하면 됨
print(ft.view([-1, 1, 3]).shape)

# Squeeze ==> 쥐어짜는 것, 자동으로 element가 1인 경우로 만들어줌
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
print(ft.squeeze())
print(ft.squeeze().shape)

# Unsqueeze ==> 원하는 dimension에 1을 더해줌
ft = torch.Tensor([0, 1, 2])
print(ft.shape)
print(ft.unsqueeze(0)) # dim=0
print(ft.unsqueeze(0).shape)
print(ft.view(1, -1))
print(ft.view(1, -1).shape)
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

# Type Casting
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())
bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())

# Concatenate <== 이어붙임
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))

# Stacking <== 쌓는다
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=1))
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

# Ones and Zeros <== 모두 1이나 0으로 만들어줌
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
print(torch.ones_like(x))
print(torch.zeros_like(x))

# In-place Operation
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.)) # x*2
print(x)
print(x.mul_(2.)) # <== 새로 선언하지 않고 기존의 값에 넣음
print(x)
