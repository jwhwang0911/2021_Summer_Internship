import torch


#Squeeze
ft = torch.FloatTensor([[0], [1], [2]]) # 3*1

print(ft.squeeze()) # 3


#Unsqueeze
ft = torch.Tensor([0, 1, 2]) # 3

print(ft.unsqueeze(0)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다. // 1*3
print(ft.unsqueeze(0).shape)