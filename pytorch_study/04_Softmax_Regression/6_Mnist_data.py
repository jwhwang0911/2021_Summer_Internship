import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    #3. 분류기 구현을 위한 사전 설정
    
    USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
    device = torch.device("cuda" if USE_CUDA else "cpu")  # GPU 사용 가능하면 사용하고 아니면 CPU 사용
    
    #Set as same seed
    random.seed(777)
    torch.manual_seed(777)
    if device == "cuda":
        torch.cuda.manual_seed_all(777)
        
    # hyperparameters
    training_epochs = 15
    batch_size = 100
    
    # GET MNIST DATA SET || if train = True, get train set and if train = False, get test set
    mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
    # if download = True, if there does not exist MNIST data, will download
    
    # dataset loader
    data_loader = DataLoader(dataset=mnist_train,
                            batch_size=batch_size, # 배치 크기는 100
                            shuffle=True,
                            drop_last=True)
    
    
    # MNIST data image of shape 28 * 28 = 784 || 
    linear = nn.Linear(784, 10, bias=True).to(device)
    
    # 비용 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수를 포함하고 있음.
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = len(data_loader)
        for x_train, y_train in data_loader: # data_loader = [(tuple)]
            x_train = x_train.view(-1,784).to(device)
            y_train = y_train.to(device)
            
            hypothesis = linear(x_train)
            cost = criterion(hypothesis,y_train)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            avg_cost += cost / total_batch
    
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        
    # 테스트 데이터를 사용하여 모델을 테스트한다.
    with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
        X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
        Y_test = mnist_test.test_labels.to(device)

        prediction = linear(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())

        # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다. r 데이터 as Tensor --> [r:r+1]
        r = random.randint(0, len(mnist_test) - 1)
        X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
        Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

        print('Label: ', Y_single_data.item())
        single_prediction = linear(X_single_data)
        print('Prediction: ', torch.argmax(single_prediction, 1).item())

        plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()