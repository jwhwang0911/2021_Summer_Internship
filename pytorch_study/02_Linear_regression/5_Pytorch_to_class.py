import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegressionModel(nn.Module):
    def __init__(self,s,t):
        super().__init__()
        self.linear = nn.Linear(s,t)
    
    def forward(self, x):
        return self.linear(x) # nn.Module class have __call__ method that can be called by calling the instance

def linear_regression(x_train,y_train,new_var,s,t):
    torch.manual_seed(1)
    
    model = LinearRegressionModel(s,t)
    
    learning_rate = 1e-5
    if(s == 1):
        learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
    
    nb_epoch = 2000
    for epoch in range(nb_epoch):
        prediction = model(x_train)
        
        cost = F.mse_loss(prediction,y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if epoch % 100 == 0:
        # 100번마다 로그 출력
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epoch, cost.item()
            ))
    pred_y = model(new_var)
    print("After training : ",pred_y)
    print("Weight, Bias : ",list(model.parameters()))

if __name__ == "__main__":
    print("Single_linear_regression : ")
    # 데이터
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])
    new_var = torch.FloatTensor([4.])    
    linear_regression(x_train,y_train,new_var,1,1)
    print("\n\n\n")
    
    
    print("Multiple_linear_regression : ")
    x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])
    y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
    new_var = torch.FloatTensor([[73,80,75]])
    linear_regression(x_train,y_train,new_var,3,1)