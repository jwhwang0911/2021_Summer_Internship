from Dataset import Data, transform, output_path, i_path, output_path, data, dataloader

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import exr

class DPCN(nn.Module):
    def __init__(self,input_channels) -> None:
        super().__init__()
        
        self.input_channels = input_channels
        
        self.layer1 = nn.Conv2d(self.input_channels, 100, kernel_size=5, padding=2,stride=1)
        self.layer2 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        self.layer3 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        self.layer4 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        self.layer5 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        self.layer6 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        self.layer7 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        self.layer8 = nn.Conv2d(100,100,kernel_size=5,padding=2,stride=1)
        self.layer9 = nn.Conv2d(100,3,kernel_size=5,padding=2,stride=1)
        self.relu = nn.ReLU(inplace=True)
        
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        torch.nn.init.xavier_uniform_(self.layer5.weight)
        torch.nn.init.xavier_uniform_(self.layer6.weight)
        torch.nn.init.xavier_uniform_(self.layer7.weight)
        torch.nn.init.xavier_uniform_(self.layer8.weight)
        torch.nn.init.xavier_uniform_(self.layer9.weight)
        
    def forward(self,x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.relu(self.layer8(x))
        x = self.layer9(x)
        
        return x
        
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
    
if __name__=="__main__":
    test_dif,y,test_spe,y_,f_albe = data.test()
    f_albe = f_albe.cuda()
    test_dif = test_dif.cuda().unsqueeze(dim=0)
    test_spe = test_spe.cuda().unsqueeze(dim=0)
    
    diffuse_model = DPCN(input_channels=27).cuda()
    specular_model = DPCN(input_channels=27).cuda()
    
    criterion = nn.L1Loss().cuda()
    diffuse_optim = optim.Adam(diffuse_model.parameters(),lr=0.00001)
    specular_optim = optim.Adam(specular_model.parameters(),lr=0.00001)
    
    epo = 0
    
    for epoch in range(epo):
        index = 0
        epo_dif_loss = 0
        epo_spe_loss = 0
        for x_dif, y_dif, x_spe, y_spe in dataloader:
            index += 1
            
            #diffuse
            x_dif = x_dif.cuda()
            y_dif = y_dif.cuda()

            diffuse_optim.zero_grad()
            dif_output = diffuse_model(x_dif)
            dif_loss = criterion(dif_output,y_dif)
            dif_loss.backward()
            epo_dif_loss += dif_loss
            diffuse_optim.step()
                
            #specular
            x_spe = x_spe.cuda()
            y_spe = y_spe.cuda()
            
            specular_optim.zero_grad()
            spe_output = specular_model(x_spe)
            spe_loss = criterion(spe_output,y_spe)
            spe_loss.backward()
            specular_optim.step()
            epo_spe_loss += spe_loss   
            
            if np.mod(index,100) == 1:    
                print('epoch {}, {}/{}, specular loss is {}'.format(epoch, index, len(dataloader), spe_loss))
                print('epoch {}, {}/{}, diffuse loss is {}'.format(epoch, index, len(dataloader), dif_loss))
        print('epoch diffuse loss = %f'%(epo_dif_loss/len(dataloader)))
        print('epoch specular loss = %f'%(epo_spe_loss/len(dataloader)))
        print()
        print()
        print()
        
        if epoch % 10 == 0:
                dif_output = diffuse_model(test_dif)
                dif_output = dif_output.mul(f_albe + 0.00316)
                dif_output = dif_output.squeeze()
                dif_output = dif_output.swapaxes(1,2).swapaxes(0,2)
                dif_output = dif_output.cpu().data.numpy().copy()
                exr.write(output_path+"diffuse/"+"dif_output_trans"+str(epoch)+"epo.exr",dif_output)
        if epoch % 10 == 0:
                spe_output = specular_model(test_spe)
                spe_output = torch.exp(spe_output)-1
                spe_output = spe_output.squeeze()
                spe_output = spe_output.swapaxes(1,2).swapaxes(0,2)
                spe_output = spe_output.cpu().data.numpy().copy()
                exr.write(output_path+"specular/"+"spe_output_trans"+str(epoch)+"epo.exr",spe_output)
                
        
    
    with torch.no_grad():
        test = Data(transform=transform, input_path = i_path)
        x_dif,y_dif,x_spe,y_spe,f_alb = test.testset()
        
        x_dif = x_dif.cuda()  
        x_spe = x_spe.cuda()
        f_alb = f_alb.cuda()
        
        x_dif = x_dif.unsqueeze(dim=0)
        x_spe = x_spe.unsqueeze(dim=0)
        
        dif_out = diffuse_model(x_dif).squeeze(dim=0)
        dif_out = dif_out.mul(f_alb + 0.00316)
        
        spe_out = specular_model(x_spe)
        spe_out = torch.exp(spe_out)-1
        spe_out = spe_out.squeeze(dim=0)
        
        y_spe = y_spe.cuda()
        y_spe = torch.exp(y_spe) - 1
        c_estimate = dif_out + spe_out
        y_spe = y_spe.cpu().data.numpy().copy()
        y_spe = y_spe.swapaxes(1,2).swapaxes(0,2)
        
        dif_out = dif_out.cpu().data.numpy().copy()       
        dif_out = dif_out.swapaxes(1,2).swapaxes(0,2)
        spe_out = spe_out.cpu().data.numpy().copy()
        spe_out = spe_out.swapaxes(1,2).swapaxes(0,2)
        c_estimate = c_estimate.cpu().data.numpy().copy() 
        c_estimate = c_estimate.swapaxes(1,2).swapaxes(0,2)
        
        exr.write(output_path+"diffuse_test.exr",dif_out)
        exr.write(output_path+"specular_test.exr",spe_out)
        exr.write(output_path+"estimate.exr",c_estimate)
        exr.write(output_path+"expect_specular.exr",y_spe)