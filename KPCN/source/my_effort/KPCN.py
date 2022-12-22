from Dataset import transform, output_path, test_path, input_path, dataloader, Data

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import exr

def apply_kernel(kernel:torch.Tensor,target:torch.Tensor, batch:int, height:int, length:int)->torch.Tensor: # kernel : batchsize * 441 * 80 * 80 , target : batchsize * 3 * 100 * 100
    result = torch.zeros(batch,3,height,length)
    for i in range(batch):
        for x in range(height):
            for y in range(length):
                subkernel = kernel[i,:,x,y].view(1,1,11,11)
                pixel = target[i:i+1,:,x:x+11,y:y+11]
                result[i,:,x,y] = (pixel * subkernel).sum(dim=2).sum(dim=2).view(3)
    
    return result


class KPCN(nn.Module):
    def __init__(self,input_channels) -> None:
        super().__init__()
        
        self.input_channels = input_channels
        
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
            nn.Conv2d(100,121,kernel_size=5,padding=2,stride=1),
            nn.Softmax(dim=1)
        )
        
        self.Model_architechture.apply(self.__weight_init__)
        
    def __weight_init__(self,sub_layer):
        if isinstance(sub_layer, nn.Conv2d):
            nn.init.xavier_uniform_(sub_layer.weight)
        
        
    def forward(self,x:torch.FloatTensor)->torch.FloatTensor:
        x = self.Model_architechture(x)
        x = F.softmax(x,dim=1)
        
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
    test_data = Data(transform=transform,input_path = test_path)
    test_dif,y,test_spe,y_,f_albe, test_target_dif, test_target_spe = test_data.test_set()
    f_albe = f_albe.cuda()
    test_dif = test_dif.cuda().unsqueeze(dim=0)
    test_spe = test_spe.cuda().unsqueeze(dim=0)
    test_target_dif = test_target_dif.cuda().unsqueeze(dim=0)
    test_target_spe = test_target_spe.cuda().unsqueeze(dim=0)
    
    diffuse_model = KPCN(input_channels=27).cuda()
    specular_model = KPCN(input_channels=27).cuda()
    
    criterion = nn.L1Loss().cuda()
    diffuse_optim = optim.Adam(diffuse_model.parameters(),lr=0.00001)
    specular_optim = optim.Adam(specular_model.parameters(),lr=0.00001)
    
    epo = 100
    
    data_num = 1
    
    for epoch in range(epo):
        index = 0
        epo_dif_loss = 0
        epo_spe_loss = 0
        
        # if epoch % 10 == 0:
        #     dif_kernel = F.softmax(diffuse_model(test_dif),dim=1)     
        #     dif_output = apply_kernel(dif_kernel,test_target_dif,1,720,1280).cuda()
        #     dif_output = dif_output.mul(f_albe + 0.00316)
        #     dif_output = dif_output.squeeze(dim=0)
        #     dif_output = dif_output.swapaxes(1,2).swapaxes(0,2)
        #     dif_output = dif_output.cpu().data.numpy().copy()
        #     exr.write(output_path+"diffuse/"+"dif_output_trans"+str(epoch)+"epo.exr",dif_output)
            
        # if epoch % 10 == 0:
        #     spe_kernel = F.softmax(specular_model(test_spe),dim=1)
        #     spe_output = apply_kernel(spe_kernel, test_target_spe,1,720,1280).cuda()
        #     spe_output = torch.exp(spe_output)-1
        #     spe_output = spe_output.squeeze(dim=0)
        #     spe_output = spe_output.swapaxes(1,2).swapaxes(0,2)
        #     spe_output = spe_output.cpu().data.numpy().copy()
        #     exr.write(output_path+"specular/"+"spe_output_trans"+str(epoch)+"epo.exr",spe_output)    
            
        
        
        for x_dif, y_dif, x_spe, y_spe, tar_dif, tar_spe in dataloader:
            index += 1
            
            #diffuse
            x_dif = x_dif.cuda()
            y_dif = y_dif.cuda()
            tar_dif = tar_dif.cuda()

            diffuse_optim.zero_grad()
            dif_kernel = diffuse_model(x_dif)
            dif_output = apply_kernel(dif_kernel,tar_dif,4,80,80).cuda()
            dif_loss = criterion(dif_output, y_dif)
            dif_loss.backward()
            epo_dif_loss += dif_loss
            diffuse_optim.step()
            
            #specular
            x_spe = x_spe.cuda()
            y_spe = y_spe.cuda()
            tar_spe = tar_spe.cuda()
            
            specular_optim.zero_grad()
            spe_kernel = specular_model(x_spe)
            spe_output = apply_kernel(spe_kernel, tar_spe, 4, 80,80).cuda()
            spe_loss = criterion(spe_output,y_spe)
            spe_loss.backward()
            epo_spe_loss += spe_loss
            specular_optim.step()
            
            
            
            if np.mod(index,10) == 1:    
                print("success")
        print('epoch diffuse loss = %f'%(epo_dif_loss/(144 * data_num)))
        print('epoch specular loss = %f'%(epo_spe_loss/(144 * data_num)))
        print()
        print()
        print()
  
    
    
    
    
    with torch.no_grad():
        test = Data(transform=transform, input_path = input_path)
        x_dif,y_dif,x_spe,y_spe,f_alb, tar_dif, tar_spe = test.test_set()
        
        tar_dif = tar_dif.cuda().unsqueeze(dim = 0)
        tar_spe = tar_spe.cuda().unsqueeze(dim = 0)
        
        x_dif = x_dif.cuda().unsqueeze(dim=0)
        tar_dif = tar_dif.cuda()
        
        x_spe = x_spe.cuda().unsqueeze(dim=0)
        tar_spe = tar_spe.cuda()
        
        dif_kernel = diffuse_model(x_dif)
        spe_kernel = specular_model(x_spe)

        diffuse = apply_kernel(dif_kernel,tar_dif,1,720,1280).squeeze(dim=0)
        specular = apply_kernel(spe_kernel,tar_spe,1,720,1280).squeeze(dim=0)
        
        diffuse = diffuse * (f_alb + 0.00316)
        specular = torch.exp(specular) - 1
        c_estimate = diffuse + specular
        
        diffuse = diffuse.cpu().data.numpy().copy()      
        diffuse = diffuse.swapaxes(1,2).swapaxes(0,2)
        specular = specular.cpu().data.numpy().copy()
        specular = specular.swapaxes(1,2).swapaxes(0,2)
        c_estimate = c_estimate.cpu().data.numpy().copy()
        c_estimate = c_estimate.swapaxes(1,2).swapaxes(0,2)
        
        exr.write(output_path+"diffuse_test.exr",diffuse)
        exr.write(output_path+"specular_test.exr",specular)
        exr.write(output_path+"estimate.exr",c_estimate)