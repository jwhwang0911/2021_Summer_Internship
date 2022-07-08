import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if __name__ == "__main__":
    torch.manual_seed(1)
    
    x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
    y_data = [[0], [0], [0], [1], [1], [1]]
    x_train = torch.FloatTensor(x_data)
    y_train = torch.FloatTensor(y_data)
    
    W = torch.zeros((2, 1), requires_grad=True) # 크기는 2 x 1
    b = torch.zeros(1, requires_grad=True)
    
    optimizer = optim.SGD([W,b],lr = 1)
    
    nb_epochs = 1000
    
    for epoch in range(nb_epochs):
        hypothesis = torch.sigmoid(x_train.matmul(W)+b)
        
        cost = -(y_train*torch.log(hypothesis)+(1-y_train)*torch.log(1-hypothesis)).mean()
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))
            
    prediction = hypothesis >= torch.FloatTensor([0.5]) # How to print if bigger then 0.5 -> True, smaller -> False
    print(prediction)

        