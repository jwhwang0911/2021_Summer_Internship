import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomDataset(Dataset): # Customize same format as TensorDataset Class
    def __init__(self):
    #데이터셋의 전처리를 해주는 부분
        self.x_data = [[73, 80, 75],
                    [93, 88, 93],
                    [89, 91, 90],
                    [96, 98, 100],
                    [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
        
    def __len__(self):
    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
        return len(self.x_data)

    def __getitem__(self, idx): 
    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
        x = torch.FloatTensor(self.x_data)
        y = torch.FloatTensor(self.y_data)
        return (x,y)


if __name__ == "__main__":
    dataset = CustomDataset()
    dataloader = DataLoader(dataset,batch_size=2,shuffle=True)
    model = torch.nn.Linear(3,1)
    
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
    
    nb_epochs = 20
    for epoch in range(nb_epochs+1):
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            
            prediction = model(x_train)
            
            cost = F.mse_loss(prediction,y_train)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
        
    # 임의의 입력 [73, 80, 75]를 선언
    new_var =  torch.FloatTensor([[73, 80, 75]])    
    # 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
    pred_y = model(new_var) 
    print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 