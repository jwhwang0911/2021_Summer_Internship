# For Deep Learning, sometimes we need to add different size tensors
# So, in pytorch, they have API that can add two tensors

import torch

#First for same size of Tensor

m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# tensor([[5., 5.]])

#Next, 1,2 and 1, Add

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] -> [3, 3]
print(m1 + m2)

# tensor([[4., 5.]]) ||| because , [3] becomes [3,3] tensor

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# tensor([4., 5.], [5., 6.]])