import os
import numpy as np

from torch.utils.data import IterableDataset, DataLoader

import torch

import random

i_path = "/rec/pvr1/deep_learning_denoising/renderings_numpy/test/128spp/"
ref_path = "/rec/pvr1/deep_learning_denoising/renderings_numpy/test/8192spp/"

test_path = "/rec/pvr1/deep_learning_denoising/renderings_numpy/test_input/128spp/"

i_dir = sorted(os.listdir(path=i_path))
ref_dir = sorted(os.listdir(path = ref_path))
test_dir = sorted(os.listdir(path=test_path))

data_num = 30

class Dataset(IterableDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data_length = data_num
        patch_size = 80
        self.X = []
        self.Y = []
        
        for idx in range(self.data_length):
            input_file : torch.Tensor = torch.from_numpy(np.load(i_path + i_dir[idx]))
            ref_file : torch.Tensor = torch.from_numpy(np.load(ref_path + ref_dir[idx]))
            print(idx, "   ", i_dir[idx] , "   ", ref_dir[idx], "   read")
            
            for j in range(1280//patch_size):
                for i in range(720//patch_size):
                    self.X.append(input_file[:,80*i:80*(i+1),80*j:80*(j+1)])
                    self.Y.append(ref_file[:,80*i:80*(i+1),80*j:80*(j+1)])                    
            
        
    def __iter__(self):
        shuffle_idx = list(range(144*self.data_length))
        random.shuffle(shuffle_idx)
        for idx in shuffle_idx:
            yield self.X[idx], self.Y[idx]
            
            
class Testset():
    def __init__(self) -> None:
        self.data_length = len(test_dir)
        self.X = []
        
        for idx in range(self.data_length):
            self.X.append(torch.from_numpy(np.load(test_path + test_dir[idx])).unsqueeze(dim=0))
    
    def testset(self):
        return self.X
                    
        
        
data = Dataset()
dataloader = DataLoader(data,batch_size=4, drop_last=True)

test = Testset()

