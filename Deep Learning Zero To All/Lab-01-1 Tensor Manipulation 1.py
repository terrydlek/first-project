# |t| = (batch size, dim) <- 2차원
# |t| = (batch size, width(length), height(dim)) <- 3차원
import numpy as np
import torch
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
print('rank of t: ', t.ndim) # 몇개의 차원?
print('shape of t', t.shape) # 형태는 어떻니?

print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

t = np.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.],[10.,11.,12.]])
print(t)
print('rank of t: ', t.ndim)
print('shape of t: ', t.shape)


t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
print(t.dim()) #rank
print(t.shape) # shape
print(t.size()) # shape
print(t[0], t[1], t[-1]) # element
print(t[2:5], t[4:-1]) # slicing
print(t[:2], t[3:]) # slicing

t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                       ])
print(t)

print(t.dim()) #rank
print(t.size()) # shape
print(t[:, 1])
print(t[:, 1].size())
print(t[:, :-1]) # 다 가져오되 맨 마지막 원소들을 제외하고 가져와

# broadcasting (자동적으로 사이즈 맞춰줌)
# same shape
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# Mul vs Matmul
print()
print('-------------')
print('Mul vs Matmul')
print('-------------')
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))

# Mean
t = torch.FloatTensor([1, 2])
print(t.mean())
# Can't use mean() on integers
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))

# sum
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))

# Max and Argmax
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.max()) # Returns one value: max
print(t.max(dim=0)) # Returns two values: max and argmax
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])
print(t.max(dim=1))
print(t.max(dim=-1))
