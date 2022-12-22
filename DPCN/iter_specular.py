import exr

from Iter_Dataset import output_path, test, dataloader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DPCN(nn.Module):
    def __init__(self,input_channels) -> None:
        super().__init__()
        
        self.input_channels = input_channels
        
        # self.layer1 = nn.Conv2d(self.input_channels, 100, kernel_size=5, padding=2,stride=1)
        # self.layer2 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        # self.layer3 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        # self.layer4 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        # self.layer5 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        # self.layer6 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        # self.layer7 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        # self.layer8 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        # self.layer9 = nn.Conv2d(100,3,kernel_size=5,padding=2,stride=1)
        # self.relu = nn.ReLU(inplace=True)
        
        self.Model_architechture = nn.Sequential(
            nn.Conv2d(self.input_channels, 100, kernel_size=5, padding=2,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100,3,kernel_size=5,padding=2,stride=1)
        )
        
        self.Model_architechture.apply(self.__weight_init__)
        
    def __weight_init__(self,sub_layer):
        if isinstance(sub_layer, nn.Conv2d):
            nn.init.xavier_uniform_(sub_layer.weight)
        
        
    def forward(self,x):
        x = self.Model_architechture(x)
        return x
        
        
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
    
if __name__ == "__main__":

    test.test_set()
    
    specular_model = DPCN(input_channels=27).cuda()
    
    criterion = nn.L1Loss().cuda()
    specular_optim = optim.Adam(specular_model.parameters(),lr=0.00001)
    
    epo = 200
    
    data_num =20
    print("# of data :",data_num)
    
    
    for epoch in range(epo):
        index = 0
        epo_dif_loss = 0
        epo_spe_loss = 0
        
        if epoch % 5 == 0:
            spe_output = specular_model(test.X_specular)
            spe_output = spe_output.squeeze()
            spe_output = torch.exp(spe_output) - 1
            spe_output = spe_output.swapaxes(1,2).swapaxes(0,2)
            spe_output = spe_output.cpu().data.numpy().copy()
            exr.write(output_path+"specular/"+"spe_output_trans"+str(epoch)+"epo.exr",spe_output)
        
        for x_dif, y_dif, x_spe, y_spe in dataloader:
            index += 1
            
            #spcular
            x_spe = x_spe.cuda()
            y_spe = y_spe.squeeze(dim = 0).cuda()
            
            specular_optim.zero_grad()
            spe_out = specular_model(x_spe)
            spe_loss = criterion(spe_out,y_spe)
            spe_loss.backward()
            epo_spe_loss += spe_loss
            specular_optim.step()
            
            if np.mod(index,100) == 1:    
                print('epoch {}, {}/{}, specular loss is {}'.format(epoch, index, 144 * data_num// 4, spe_loss))
        print('epoch specular loss = %f'%(epo_spe_loss/(144 * data_num)))
        print()
        print()
        print()   
        
    
    with torch.no_grad():        
        spe_out = specular_model(test.X_specular).squeeze(dim=0)
        spe_out = torch.exp(spe_out) - 1
        
        spe_out = spe_out.cpu().data.numpy().copy()       
        spe_out = spe_out.swapaxes(1,2).swapaxes(0,2)       
        
        exr.write(output_path+"specular_test.exr",spe_out)