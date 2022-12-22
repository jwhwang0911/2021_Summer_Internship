import os
import numpy as np

from torch.utils.data import IterableDataset, DataLoader
from gradient import gradient_exr

import torch
import torch.nn as nn

import random
import exr
# from torchvision import transforms

input_path = "/rec/pvr1/deep_learning_denoising/renderings_numpy/test/"
output_path = "/rec/pvr1/deep_learning_denoising/KPCN/KPCN_test/"
test_path = "/rec/pvr1/deep_learning_denoising/renderings_numpy/test_input/"

# transform = transforms.Compose([transforms.ToTensor()
#                                 ])
  

class Data(IterableDataset):    
    def __init__(self,transform = None,input_path = input_path) -> None:
        super().__init__()
        self.input_path = input_path
        self.raw_data_dir = sorted(os.listdir(path=input_path))
        self.transform = transform
        self.patchsize = 80
        # self.data_length = 1
        self.data_length = min(len(self.raw_data_dir)//5,20)
        
        self.f_albe = None
        
        self.X_diffuse = []
        self.Y_diffuse = []
        self.X_specular = []
        self.Y_specular = [] 
        
        for i in range(self.data_length):
            self.preprocessing(i)
    
    def len_data(self):
        return self.data_length
    
    def preprocessing(self, file_index, train = True):
        input_image = self.raw_data_dir[file_index*5 + 0]
        reference_image = self.raw_data_dir[file_index*5 + 4]
        print(str(file_index)+"th image, "+"read : ",input_image, reference_image)
        
        epsilon = 0.00316
        
        input_file = np.load(self.input_path + input_image,allow_pickle=True).item()
        reference_file = np.load(self.input_path + reference_image,allow_pickle=True).item()
        
        # reference processing
          # diffuse
        c_diffuse_ref = torch.from_numpy(reference_file["diffuse"]).swapaxes(0,2).swapaxes(1,2)
        f_albedo_ref = torch.from_numpy(reference_file["albedo"]).swapaxes(0,2).swapaxes(1,2)
        
        Y_diffuse = torch.div(c_diffuse_ref, (f_albedo_ref+epsilon))
        
          # specular
        c_specular_ref = torch.from_numpy(reference_file["specular"]).swapaxes(0,2).swapaxes(1,2)
        c_specular_ref = torch.clamp(c_specular_ref, min=0.0)
        
        Y_specular = torch.log(1+c_specular_ref)
        
        # X processing
        
        c_diffuse = torch.from_numpy(input_file["diffuse"]).swapaxes(0,2).swapaxes(1,2)
        c_diffuse_var = torch.from_numpy(input_file["diffuseVariance"]).swapaxes(0,2).swapaxes(1,2)
        
        c_specular = torch.from_numpy(input_file["specular"]).swapaxes(0,2).swapaxes(1,2)
        c_specular_var = torch.from_numpy(input_file["specularVariance"]).swapaxes(0,2).swapaxes(1,2)

        f_normal = torch.from_numpy(input_file["normal"]).swapaxes(0,2).swapaxes(1,2)
        f_albedo = torch.from_numpy(input_file["albedo"]).swapaxes(0,2).swapaxes(1,2)
        f_depth = torch.from_numpy(input_file["depth"]).swapaxes(0,2).swapaxes(1,2)
        
        max_depth = f_depth.max()
        f_depth = torch.div(f_depth,max_depth)

        f_normal_var = torch.from_numpy(input_file["normalVariance"]).swapaxes(0,2).swapaxes(1,2)
        f_albedo_var = torch.from_numpy(input_file["albedoVariance"]).swapaxes(0,2).swapaxes(1,2)
        f_depth_var = torch.from_numpy(input_file["depthVariance"]).swapaxes(0,2).swapaxes(1,2)

        c_diffuse_trans = torch.div(c_diffuse, (f_albedo + epsilon))
        c_diffuse_var_trans = torch.div(c_diffuse_var, ((f_albedo + epsilon)*(f_albedo + epsilon)).sum(dim=0).unsqueeze(dim=0))

        c_specular_trans= torch.clamp(c_specular, min = 0.0)
        c_specular_trans = torch.log(c_specular+1)
        #######
        # c_specular_trans = c_specular

        c_specular_var_trans = torch.div(c_specular_var, ((c_specular_trans+epsilon)*(c_specular_trans+epsilon)).sum(dim=0).unsqueeze(dim=0))
        ########
        # c_specular_var_trans = c_specular_var


        Gx_diffuse, Gy_diffuse = gradient_exr(c_diffuse_trans)
        Gx_specular, Gy_specular = gradient_exr(c_specular_trans)

        Gx_normal, Gy_normal = gradient_exr(f_normal)
        Gx_albedo, Gy_albedo = gradient_exr(f_albedo)
        Gx_depth, Gy_depth = gradient_exr(f_depth)
        
        X_diffuse = torch.cat((c_diffuse_trans,Gx_diffuse, Gx_albedo, Gx_depth,
                                  Gx_normal, Gy_diffuse, Gy_albedo, Gy_depth, Gy_normal,
                                  c_diffuse_var_trans, f_normal_var, f_albedo_var
                                  ,f_depth_var
                                  ),dim=0)
        
        X_specular = torch.cat((c_specular_trans,Gx_specular, Gx_albedo, Gx_depth,
                          Gx_normal, Gy_specular, Gy_albedo, Gy_depth, Gy_normal,
                          c_specular_var_trans, f_normal_var, f_albedo_var
                          ,f_depth_var
                          ),dim=0)
        
        X_diffuse = torch.nan_to_num(X_diffuse, nan=0.0, posinf=float(X_diffuse.max()), neginf= 0.0 )
        X_specular = torch.nan_to_num(X_specular, nan=0.0, posinf=float(X_diffuse.max()), neginf= 0.0 )
        Y_diffuse = torch.nan_to_num(Y_diffuse, nan=0.0, posinf=float(X_diffuse.max()), neginf= 0.0 )
        Y_specular = torch.nan_to_num(Y_specular, nan=0.0, posinf=float(X_diffuse.max()), neginf= 0.0 )
        
        if not train:
            self.X_diffuse = X_diffuse
            self.Y_diffuse = Y_diffuse
            self.X_specular = X_specular
            self.Y_specular = Y_specular
            self.f_albe = f_albedo
            return
        
        
        for j in range(1280//self.patchsize):
            for i in range(720//self.patchsize):
                self.X_diffuse.append(X_diffuse[:,80*i:80*(i+1),80*j:80*(j+1)])
                self.Y_diffuse.append(Y_diffuse[:,80*i:80*(i+1),80*j:80*(j+1)])        
                self.X_specular.append(X_specular[:,80*i:80*(i+1),80*j:80*(j+1)])
                self.Y_specular.append(Y_specular[:,80*i:80*(i+1),80*j:80*(j+1)]) 

        
    def __iter__(self):
        shuffle_idx = list(range(144*self.data_length))
        random.shuffle(shuffle_idx)
        for idx in shuffle_idx:
            yield self.X_diffuse[idx], self.Y_diffuse[idx], self.X_specular[idx], self.Y_specular[idx]
        
    
    def test_set(self):
        self.preprocessing(0,train=False) 
        self.X_diffuse = self.X_diffuse.cuda().unsqueeze(dim = 0)
        self.X_specular = self.X_specular.cuda().unsqueeze(dim = 0)
        self.f_albe = self.f_albe.cuda()
   
    
data = Data(transform=None, input_path=input_path)
dataloader = DataLoader(data,batch_size=4,drop_last=True)

test = Data(transform=None, input_path = test_path)

## Test of target_processing

# temp = Data(transform,test_path)
# x_dif, Y_dif, x_spe, y_spe, f_albe, tar_dif, tar_spe = temp.test_set()
# tar_dif = tar_dif.unsqueeze(dim = 0)
# tar_spe = tar_spe.unsqueeze(dim = 0)

# tar_dif_r, tar_dif_g, tar_dif_b = target_processing(tar_dif)
# tar_dif_r, tar_dif_g, tar_dif_b = tar_dif_r.mean(dim=1), tar_dif_g.mean(dim=1), tar_dif_b.mean(dim=1)

# diffuse = torch.cat((tar_dif_r, tar_dif_g, tar_dif_b),dim=0)

# tar_spe_r, tar_spe_g, tar_spe_b = target_processing(tar_spe)
# tar_spe_r, tar_spe_g, tar_spe_b = tar_spe_r.mean(dim=1), tar_spe_g.mean(dim=1), tar_spe_b.mean(dim=1)

# specular = torch.cat((tar_spe_r, tar_spe_g, tar_spe_b),dim=0)

# c_estimate = diffuse.mul(f_albe + 0.00316) + torch.exp(specular) - 1

# c_estimate = c_estimate.cpu().data.numpy().copy() 
# c_estimate = c_estimate.swapaxes(1,2).swapaxes(0,2)


# exr.write(output_path+"filter.exr",c_estimate)




# def apply_kernel(target:torch.Tensor, batch:int, height:int, length:int)->torch.Tensor: # kernel : batchsize * 441 * 80 * 80 , target : batchsize * 3 * 100 * 100
#     kernel = torch.ones(batch,441,height,length)
#     result = torch.zeros(batch,3,height,length)
#     j = 0
#     for i in range(batch):
#         for x in range(height):
#             for y in range(length):
#                 subkernel = kernel[i,:,x,y].view(1,1,21,21)
#                 pixel = target[i:i+1,:,x:x+21,y:y+21]
#                 result[i,:,x,y] = (pixel * subkernel).sum(dim=2).sum(dim=2).view(3)/441
    
#     return result

# ## Test of apply_kernel

# temp = Data(transform,test_path)
# x_dif, Y_dif, x_spe, y_spe, f_albe, tar_dif, tar_spe = temp.test_set()
# tar_dif = tar_dif.unsqueeze(dim = 0)
# tar_spe = tar_spe.unsqueeze(dim = 0)

# print(tar_dif.shape)
# print(tar_spe.shape)

# diffuse = apply_kernel(tar_dif,1,720,1280)
# specular = apply_kernel(tar_spe,1,720,1280)

# print(diffuse.shape)
# print(specular.shape)

# c_estimate = diffuse.mul(f_albe + 0.00316) + torch.exp(specular) - 1
# print(c_estimate.shape)
# c_estimate = c_estimate.squeeze(dim=0)
# c_estimate = c_estimate.cpu().data.numpy().copy() 
# c_estimate = c_estimate.swapaxes(1,2).swapaxes(0,2)
# print(c_estimate.shape)


# exr.write(output_path+"filter.exr",c_estimate)