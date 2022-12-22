from gradient import gradient_exr
import numpy as np
import exr
import os
import sys
import torch


exr_path = "/rec/pvr1/deep_learning_denoising/renderings/test/"
input_path = "/rec/pvr1/deep_learning_denoising/renderings_numpy/test/128spp/"
refer_path = "/rec/pvr1/deep_learning_denoising/renderings_numpy/test/8192spp/"


def preprocessing(file_name: str, x : dict) -> torch.Tensor:
    if not isinstance(x,dict):
        sys.exit("File incorrect : "+file_name)
    
    epsilon = 0.00316
    
    albedo = torch.from_numpy(x["albedo"]).swapaxes(0,2).swapaxes(1,2)
    normal = torch.from_numpy(x["normal"]).swapaxes(0,2).swapaxes(1,2)
    depth = torch.from_numpy(x["depth"]).swapaxes(0,2).swapaxes(1,2)
    
    albedo_var = torch.from_numpy(x["albedoVariance"]).swapaxes(0,2).swapaxes(1,2)
    normal_var = torch.from_numpy(x["normalVariance"]).swapaxes(0,2).swapaxes(1,2)
    depth_var = torch.from_numpy(x["depthVariance"]).swapaxes(0,2).swapaxes(1,2)
    
    Gx_albedo, Gy_albedo = gradient_exr(albedo)
    Gx_normal, Gy_normal = gradient_exr(normal)
    Gx_depth, Gy_depth = gradient_exr(depth)
    
    
    c_diffuse = torch.div(torch.from_numpy(x["diffuse"]).swapaxes(0,2).swapaxes(1,2), (albedo + epsilon))
    c_diffuse_var = torch.div(torch.from_numpy(x["diffuseVariance"]).swapaxes(0,2).swapaxes(1,2), ((albedo + epsilon) * (albedo + epsilon)).sum(dim=0).unsqueeze(dim=0))
    c_specular = torch.log(torch.from_numpy(x["specular"]).swapaxes(0,2).swapaxes(1,2) + 1)
    c_specular_var = torch.div(torch.from_numpy(x["specularVariance"]).swapaxes(0,2).swapaxes(1,2), ((c_specular + epsilon) * (c_specular + epsilon)).sum(dim=0).unsqueeze(dim=0) )
    
    Gx_diffuse, Gy_diffuse = gradient_exr(c_diffuse)
    Gx_specular, Gy_specular = gradient_exr(c_specular)
    
    
    return torch.cat((c_diffuse, Gx_diffuse, Gy_diffuse, c_diffuse_var,
                      Gx_albedo, Gx_depth, Gx_normal, Gy_albedo, Gy_depth, Gy_normal,
                      albedo_var, normal_var, depth_var,
                      c_specular, Gx_specular, Gy_specular, c_specular_var, albedo
                      ), dim=0)
    
    # diffuse = idx 0 ~ 26
    # specular = idx 10 ~ 36
    # f_albedo = idx 37 ~ 39
    
def ref_preprocessing(file_name: str, x : dict) -> torch.Tensor:
    if not isinstance(x,dict):
        sys.exit("File incorrect : "+file_name)
    
    epsilon = 0.00316
    
    albedo = torch.from_numpy(x["albedo"]).swapaxes(0,2).swapaxes(1,2)
    
    
    c_diffuse = torch.div(torch.from_numpy(x["diffuse"]).swapaxes(0,2).swapaxes(1,2), (albedo + epsilon))
    c_specular = torch.log(torch.from_numpy(x["specular"]).swapaxes(0,2).swapaxes(1,2) + 1)
    
    
    return torch.cat((c_diffuse,
                      c_specular,
                      albedo
                      ), dim=0)
    

if __name__ == "__main__":
    exr_files = sorted(os.listdir(exr_path))

    if os.listdir("/rec/pvr1/deep_learning_denoising/renderings_numpy/test_input/") is []:
        os.mkdir(input_path)
        os.mkdir(refer_path)
    
    for file_idx in range(len(exr_files)):
        mod = file_idx % 5
        if mod == 0 or mod == 4:
            file_name = exr_files[file_idx]
            t = exr.read_all(exr_path + file_name)
            file_name, _ = file_name.split(".")
            
            
            if mod == 0:
                temp = preprocessing(file_name, t)
                print(temp.shape)
                temp = temp.numpy()
                np.save(input_path + file_name + ".npy",temp)
                
            if mod == 4:
                temp = ref_preprocessing(file_name, t)
                print(temp.shape)
                temp = temp.numpy()
                np.save(refer_path + file_name + ".npy" , temp)
            
            
            print(file_name, "fin")
            
        
            