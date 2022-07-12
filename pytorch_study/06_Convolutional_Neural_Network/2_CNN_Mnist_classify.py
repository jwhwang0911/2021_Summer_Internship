from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        
        self.layer1 = nn.Sequential(nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        
        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        
        self.layer2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        
        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = nn.Linear(7*7*64,10,bias=True)
        
        # 전결합층 한정으로 가중치 초기화
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
        
        

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(777)
    
    if device == "cuda":
        torch.cuda.manual_seed_all(777)
        
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100
    
    mnist_train = dsets.MNIST(root="MNIST_data/",
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True
                              )
    
    mnist_test = dsets.MNIST(root="MNIST_data/",
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True
                             )
    
    data_loader = DataLoader(dataset=mnist_train,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True 
                            )
    
    model = CNN().to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    
    total_batch = len(data_loader)
    print('총 배치의 수 : {}'.format(total_batch))
    
    for epoch in range(training_epochs):
        avg_cost = 0
        
        for X,Y in data_loader:
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis,Y)
            cost.backward()
            optimizer.step()
            
            avg_cost += cost/total_batch
            
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
        
    # 학습을 진행하지 않을 것이므로 torch.no_grad()
    with torch.no_grad():
        X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        Y_test = mnist_test.test_labels.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())