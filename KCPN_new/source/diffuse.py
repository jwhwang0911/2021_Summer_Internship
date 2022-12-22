import exr
from Dataset import dataloader, test, data_num

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.cpp_extension import load
print(torch.__version__)


output_path = "/rec/pvr1/deep_learning_denoising/KPCN_new/test/"
# Load custom module
print("here")
ops = load(name='weithed_average', sources=['ops.cu'], verbose=True)


class WeightedAverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input, weights)
        output = ops.forward(input, weights)
        return output

    @staticmethod
    def backward(ctx, gradPrev):
        input, weights = ctx.saved_tensors
        grads = ops.backward(input, weights, gradPrev)

        return None, grads[1]
    
class KPCN(torch.nn.Module):
    def __init__(self, kernelWidth,input_channels):
        super(KPCN, self).__init__()
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
            nn.Conv2d(100,kernelWidth*kernelWidth,kernel_size=5,padding=2,stride=1),
            nn.Softmax(dim=1)
        )
        
        self.Model_architechture.apply(self.__weight_init__)

    def __weight_init__(self,sub_layer):
        if isinstance(sub_layer, nn.Conv2d):
            nn.init.xavier_uniform_(sub_layer.weight)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        w = self.Model_architechture(x)
        out = []
        
        for i in range(len(x)):
            out.append(WeightedAverage.apply(x[i,0:3,:,:],w[i]).unsqueeze(dim=0))
    
        return torch.cat(tuple(out),dim=0)
    
if __name__ == "__main__":
    test_list = test.testset()
    
    diffuse_model = KPCN(kernelWidth = 21,input_channels=27).cuda()
    criterion = nn.L1Loss().cuda()
    diffuse_optim = optim.Adam(diffuse_model.parameters(),lr=0.00001)
    
    epo = 200
    
    print("# of data :",data_num)
    
    for epoch in range(epo):
        index = 0
        epo_dif_loss = 0
        epo_spe_loss = 0
        
        if epoch % 10 == 0:
            _ = 1
            for e in test_list:
                dif_input =e[:,0:27,:,:].cuda()
                dif_output = diffuse_model(dif_input)
                f_albe = e[:,37:40,:,:].cuda()
                dif_output = dif_output * (f_albe + 0.00316)
                dif_output = dif_output.squeeze(dim=0)
                dif_output = dif_output.swapaxes(1,2).swapaxes(0,2)
                dif_output = dif_output.cpu().data.numpy().copy()
                exr.write(output_path+"diffuse/"+str(epoch)+"/dif_output_trans_"+str(_)+"th.exr",dif_output)
                _ += 1
        
        for x_dif, y_dif in dataloader:
            index += 1
            
            #diffuse
            x_dif = x_dif[:,0:27,:,:].cuda()
            y_dif = y_dif[:,0:3,:,:].cuda()

            diffuse_optim.zero_grad()
            dif_out = diffuse_model(x_dif)
            dif_loss = criterion(dif_out,y_dif)
            dif_loss.backward()
            epo_dif_loss += dif_loss
            diffuse_optim.step()
            
            if np.mod(index,30) == 1:    
                print('epoch {}, {}/{}, diffuse loss is {}'.format(epoch, index, 144 * data_num// 4, dif_loss))
        print('epoch diffuse loss = %f'%(epo_dif_loss/(144 * data_num)))
        print()
        print()
        print() 
        
    with torch.no_grad():
        for e in test_list:
            dif_input = e[:,0:27,:,:].cuda()
            dif_output = diffuse_model(dif_input)
            dif_output = dif_output.squeeze()
            f_albe = e[:,37:40,:,:].cuda()
            dif_output = dif_output * (f_albe + 0.00316)
            dif_output = dif_output.swapaxes(1,2).swapaxes(0,2)
            dif_output = dif_output.cpu().data.numpy().copy()
            exr.write(output_path+"diffuse.exr",dif_output)