import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if __name__ == "__main__":
    # 현재 실습하고 있는 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 줍니다.
    torch.manual_seed(1)
    
    #train set
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])
    
    # Weight W, bias b를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함.
    W = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    # optimizer 설정
    optimizer = optim.SGD([W, b], lr=0.01)
    # 'SGD'는 경사 하강법의 일종

    
    #For know, y = 0*x+b
    
    nb_epochs = 1999 # number that want to use 경사 하강법
    
    for epoch in range(nb_epochs + 1):
        #H(x)
        hypothesis = W * x_train + b
        
        #Costs function
        cost = torch.mean((hypothesis - y_train)**2)
        
        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
            
        # 100번마다 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            ))
    