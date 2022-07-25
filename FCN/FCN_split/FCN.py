import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from FCN_split.GetData import GetDataSet, transform

from torch.utils.data import DataLoader

import FCN_split.exr as exr
import numpy as np
import time

class FCN32s(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        
        #Pool number 0
        self.layer1_1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64)
                                    )
        
        self.layer1_2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d(2,2)
                                    )
        
        #Pool number 1
        self.layer2_1 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128)
                                    )        
         
        self.layer2_2 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.MaxPool2d(2,2)
                                    )
        
        #Pool number 2
        self.layer3_1 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256)
                                    )
        
        self.layer3_2 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256)
                                    )
        
        self.layer3_3 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.MaxPool2d(2,2)
                                    )
        #Pool number 3
        self.layer4_1 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(512)
                                    )
        
        self.layer4_2 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(512)
                                    )
        
        self.layer4_3 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3, padding=1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(512),
                                    nn.MaxPool2d(2,2)
                                    )
        
        #Pool number 4

        #Transpose Convolution
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        
    def forward(self,x):
        #Convolution
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x1 = x
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x2 = x
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x3 = x
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.layer4_3(x)
        
        #Transpose Convolution
        x = self.bn1(self.relu(self.deconv1(x)))
        x = x + x3
        x = self.bn2(self.relu(self.deconv2(x)))
        x = x + x2
        x = self.bn3(self.relu(self.deconv3(x)))
        x = x + x1
        x = self.bn4(self.relu(self.deconv4(x)))
        
        x = self.classifier(x)
        
        return x
    
if __name__ == "__main__":  
    
    output_list = []
    exr_output = []
     
    for idx in range(0,9*16):
        fcn_model = FCN32s(n_class=3)
        fcn_model = fcn_model.cuda()
        criterion = nn.MSELoss().cuda()
        optimizer = optim.Rprop(fcn_model.parameters(), lr=1e-3)
        bag = GetDataSet(transform=transform,index=idx)
        dataloader = DataLoader(bag,batch_size=1)
        for epo in range(50):
            index = 0
            epo_loss = 0
            start = time.time()
            for item in dataloader:
                index += 1
                start = time.time()
                input = item['A']
                y = item['B']

                input = input.cuda()
                y = y.cuda()

                optimizer.zero_grad()
                output = fcn_model(input)
                output = F.softplus(output)
                loss = criterion(output, y)
                loss.backward()
                iter_loss = loss
                epo_loss += iter_loss
                optimizer.step()
            print('epoch loss = %f'%(epo_loss/len(dataloader)))
        print("idx : ",idx)
        
        with torch.no_grad():        
            for item in dataloader:
                input = item['A'].cuda()
                output = fcn_model(input)
                output = F.softplus(output) 
                output = output.squeeze()
                output_np = output.cpu().data.numpy().copy()
                output_np = output_np.swapaxes(1,2).swapaxes(0,2)  
                output_list.append(output_np)
    
    elem_output = output_list[0]
    
    for i in range(1,len(output_list)):
            if i % 9 == 0:
                exr_output.append(elem_output)
                elem_output = output_list[i]
            else:
                elem_output = np.concatenate((elem_output,output_list[i]),axis=0)
    
    final_output = exr_output[0]        
    
    for e_idx in range(1,len(exr_output)):
            final_output = np.concatenate((final_output,exr_output[e_idx]),axis = 1)
            
    exr.write("/rec/pvr1/deep_learning_denoising/test_write/"+"test_output.exr",
                final_output
                )

        
        