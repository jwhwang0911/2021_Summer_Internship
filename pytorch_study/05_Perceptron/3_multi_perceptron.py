import torch
import torch.nn as nn

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(777)
    if device == "cuda":
        torch.cuda.manual_seed_all(777)
        
    X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device)
    Y = torch.FloatTensor([[0],[1],[1],[0]]).to(device)
    
    model = nn.Sequential(
        nn.Linear(2,10,bias = True),
        nn.Sigmoid(),
        nn.Linear(10,10,bias = True),
        nn.Sigmoid(),
        nn.Linear(10,10,bias = True),
        nn.Sigmoid(),
        nn.Linear(10,1,bias = True),
        nn.Sigmoid()
    ).to(device)
    
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr = 1)
    
    for epoch in range(10001):
        hypothesis = model(X)
        cost = criterion(hypothesis,Y)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        
        # 100의 배수에 해당되는 에포크마다 비용을 출력
        if epoch % 100 == 0:
            print(epoch, cost.item())
            
    with torch.no_grad():
        hypothesis = model(X)
        predicted = (hypothesis > 0.5).float()
        accuracy = (predicted == Y).float().mean()
        print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
        print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
        print('실제값(Y): ', Y.cpu().numpy())
        print('정확도(Accuracy): ', accuracy.item())
        