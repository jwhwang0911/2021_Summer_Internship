from numpy import pad
import torch
import torch.nn as nn
import torch.nn.init

import torchvision.datasets as dsets
import torchvision.transforms as transform

from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.keep_prob = 0.5
        
        self.layer1 = nn.Sequential(nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.layer2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.layer3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2,padding=1))
        
        self.fc1 = nn.Linear(4*4*128,625,bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        
        self.layer4 = torch.nn.Sequential(self.fc1,
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(p=1 - self.keep_prob))
        
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.layer4(out)
        out = self.fc2(out)
        
        return out

if __name__=="__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(777)
    
    if device == "cuda":
        torch.cuda.manual_seed_all(777)
    
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100
    
    mnist_train = dsets.MNIST(root="MNIST_data/",
                              transform=transform.ToTensor(),
                              train=True,
                              download=True
                              )
    
    mnist_test = dsets.MNIST(root = "MNIST_data/",
                             train=False,
                             transform=transform.ToTensor(),
                             download=True)
    
    data_loader = DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)
    total_batch = len(data_loader)
    
    model = CNN().to(device)
    
    criterian = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    for epoch in range(training_epochs):
        avg_cost = 0
        for X,Y in data_loader:
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterian(hypothesis,Y)
            cost.backward()
            optimizer.step()
            
            avg_cost += cost / total_batch
            
        print('[Epoch : {:>4}],cost={:>.9}'.format(epoch+1,avg_cost))
        
    with torch.no_grad():
        X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        Y_test = mnist_test.test_labels.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
            
    