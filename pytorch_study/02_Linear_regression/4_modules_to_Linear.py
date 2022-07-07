#파이토치에서는 선형 회귀 모델이 nn.Linear()라는 함수
#평균 제곱오차가 nn.functional.mse_loss()라는 함수

import torch
import torch.nn as nn
import torch.nn.functional as F

def Single_linear_regression():
    # 데이터
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])
    
    model = nn.Linear(1,1) # parameter : input and output's dimension
    # Randomly initailized W and b
    
    learaning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(),lr = learaning_rate) # model.parametes return
    
    nb_epochs = 3000
    for epoch in range(nb_epochs+1):
        prediction = model(x_train) #Linear object parameter <-- training set
        
        cost = F.mse_loss(prediction,y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            # 100번마다 로그 출력
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))
            
    new_var = torch.FloatTensor([4.])
    pred_y = model(new_var)
    print("After training : ",pred_y)
    print("Weight, Bias : ",list(model.parameters()))
    
def Multi_linear_regression():
    x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
    y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
    
    model = nn.Linear(3,1)
    
    learning_rate = 1e-5
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    
    nb_epochs = 3000
    for epoch in range(nb_epochs+1):
        predication = model(x_train)
        
        cost = F.mse_loss(y_train,predication)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            # 100번마다 로그 출력
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))
            
    new_var = torch.FloatTensor([[73,80,75]])
    pred_y = model(new_var)
    print("After Training : ",pred_y)

if __name__ == "__main__":
    torch.manual_seed(1)
    print("Single_linear_regression : ")
    Single_linear_regression()
    print("\n\n\n")
    print("Multiple_linear_regression : ")
    Multi_linear_regression()