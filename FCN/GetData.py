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

# print(type(exr.read(path+sorted_dir[0]))) | numpy.ndarray

class GetDataset(Dataset):
    def __init__(self,transform=None):
        idx = 0
        self.input_image_name = sorted_dir[idx//5*5]
        self.output_image_name = sorted_dir[idx//5*5 + 4]
        self.x = exr.read(path+self.input_image_name)
        self.y = exr.read(path+self.output_image_name)
        
        self.max_height_index = size(self.x[:,0,0])//80
        self.max_width_index = size(self.x[0,:,0])//80

        
        self.transform = transform
        
    def __len__(self):
        return self.max_height_index * self.max_width_index
    
    def __getitem__(self, index):
        
        i = index//9
        j = index%9
        
        input_x = self.x[80*j:80*(j+1),80*i:80*(i+1),:]
        output_y = self.y[80*j:80*(j+1),80*i:80*(i+1),:] 

        output_y = output_y.swapaxes(0, 2).swapaxes(1, 2)        
        if self.transform:
            input_x = self.transform(input_x)
        output_y = torch.FloatTensor(output_y)
        
        item = {'A':input_x, 'B':output_y}
        
        return item
        
class GetFullData(Dataset):
    def __init__(self,transform=None):
        idx = 0
        self.input_image_name = sorted_dir[idx//5*5]
        self.output_image_name = sorted_dir[idx//5*5 + 4]
        self.x = exr.read(path+self.input_image_name)
        self.y = exr.read(path+self.output_image_name)
        
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(path))//5

    def __getitem__(self, idx):
        input = self.x.swapaxes(0, 2).swapaxes(1, 2)
        output = self.y.swapaxes(0, 2).swapaxes(1, 2) 
        input = torch.FloatTensor(input)
        output = torch.FloatTensor(output)   
        
        item = {'A':input, 'B':output}
        
        return item
        
        
bag = GetDataset(transform)
dataloader = DataLoader(bag, batch_size=4, shuffle=True, num_workers=4,drop_last=True)

full_data = DataLoader(GetFullData(transform),batch_size = 1,shuffle=True,drop_last=True)

if __name__ =='__main__':
    for batch in dataloader:
        break

    print(len(dataloader))