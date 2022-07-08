import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib as plt
import random

class MNITclass(nn.Module):
    def __init__(self):
        super().__init__
        self.linear == nn.Linear(28*28,10,bias=True)
        
    def forward(self,x):
        return self.linear(x)

if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    random.seed(777)
    torch.manual_seed(777)
    if device == "cuda":
        torch.cuda.manual_seed_all(777)
        
    training_epochs = 15
    batch_size = 100
        
    # MNIST dataset
    mnist_train = dsets.MNIST(root='MNIST_data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

    dataloader = DataLoader(dataset= mnist_train,shuffle=True,batch_size=batch_size,drop_last=True)

    linear = MNITclass().to(device)

    optimizer = torch.optim.SGD(linear.parameters(),lr=0.1)
    criterian = nn.CrossEntropyLoss().to(device)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = len(dataloader)
        for x_train, y_train in dataloader:
            x_train = x_train.view(-1,28*28).to(device)
            y_train = y_train.to(device)
            
            hypothesis = linear(x_train)
            cost = criterian(hypothesis,y_train)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            avg_cost += cost / total_batch
        
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning finished')



