import torch

t = torch.FloatTensor([[1, 2], [3, 4]])

print(t.max())
# tensor(4.)

print(t.max(dim=0))
# (tensor([3., 4.]), tensor([1, 1])) -> last is for index