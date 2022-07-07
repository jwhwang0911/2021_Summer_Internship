import torch
import torch.optim as optim

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

W = torch.zeros((3,1),requires_grad=True)
B = torch.zeros(1,requires_grad=True)

learning_rate= 1e-5
optimizer = optim.SGD([W,B],lr = learning_rate)

nb_epochs = 20

for epochs in range(nb_epochs):
    hypothesis = x_train.matmul(W) + B
    
    costs = torch.mean((hypothesis-y_train)**2)
    
    optimizer.zero_grad()
    costs.backward()
    optimizer.step()
    
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epochs, nb_epochs, hypothesis.squeeze().detach(), costs.item()
    ))