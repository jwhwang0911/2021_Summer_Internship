from operator import index
from numpy import size
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import pdb
import torch

import exr

path = "/rec/pvr1/deep_learning_denoising/renderings/test/"
sorted_dir = sorted(os.listdir(path))

transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])

class GetDataSet(Dataset):
    def __init__(self,transform = None, index=0):
        self.input_image_name = sorted_dir[0//5*5]
        self.output_image_name = sorted_dir[0//5*5 + 4]
        
        self.x = exr.read(path+self.input_image_name)
        self.y = exr.read(path+self.output_image_name)  
        
        i = index//9
        j = index%9
        
        self.x = self.x[80*j:80*(j+1),80*i:80*(i+1),:]
        self.y = self.y[80*j:80*(j+1),80*i:80*(i+1),:]
        
        self.y = self.y.swapaxes(0, 2).swapaxes(1, 2)
        
        self.transform = transform     
        
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        if self.transform:
            input_x = self.transform(self.x)
        output_y = torch.FloatTensor(self.y)
        
        return {'A':input_x,'B':output_y}
