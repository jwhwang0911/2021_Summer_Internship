import os

import exr
from gradient import gradient_exr
from torch.utils.data import Dataset, DataLoader

import torch
from torchvision import transforms

i_path = "/rec/pvr1/deep_learning_denoising/renderings/test/"
output_path = "/rec/pvr1/deep_learning_denoising/DPCN/DPCN_test/"
test_path = "/rec/pvr1/deep_learning_denoising/renderings/test_input/"

transform = transforms.Compose([
    transforms.ToTensor()
])

class Data(Dataset):
    def __init__(self, transform=None, input_path = i_path) -> None:
      self.input_path = input_path
      self.raw_data_dir = sorted(os.listdir(path=input_path))
      self.transform = transform
      self.file_len = 5
      
      self.X_diffuse = None
      self.Y_diffuse = None
      self.X_specular = None
      self.Y_specular = None
      self.f_albedo = None
      
      temp_x_dif = None
      temp_y_dif = None
      temp_x_spe = None
      temp_y_spe = None
      temp_f_albe = None
      for idx in range(self.file_len//5):
        self.preprocessing(idx)
        if idx is 0:
          temp_x_dif = self.X_diffuse
          temp_y_dif = self.Y_diffuse
          temp_x_spe = self.X_specular
          temp_y_spe = self.Y_specular
          temp_f_albe = self.f_albedo
          continue
        temp_x_dif = torch.cat((temp_x_dif,self.X_diffuse),dim=1)
        temp_y_dif = torch.cat((temp_y_dif,self.Y_diffuse),dim=1)
        temp_x_spe = torch.cat((temp_x_spe,self.X_specular),dim=1)
        temp_y_spe = torch.cat((temp_y_spe,self.Y_specular),dim=1)
        temp_f_albe = torch.cat((temp_f_albe,self.f_albedo),dim=1)
        
      self.X_diffuse = temp_x_dif
      self.Y_diffuse = temp_y_dif
      self.X_specular = temp_x_spe
      self.Y_specular = temp_y_spe
      self.f_albedo = temp_f_albe
        
    def __len__(self)->int:
      return self.file_len * 9 * 16 // 5
      # return 9 * 16
      # return 1
    
    def preprocessing(self,file_index):
        
      input_image = self.raw_data_dir[file_index*5 + 0]
      output_image = self.raw_data_dir[file_index*5 + 4]
      
      print(str(file_index)+"th image, "+"read : ",input_image, output_image)
      
      epsilon = 0.00316
      
      exr_input_file = exr.read_all(self.input_path + input_image)
      exr_output_file = exr.read_all(self.input_path + output_image)
      
      # reference(Y) processing
        # diffuse
      c_diffuse_out = self.transform(exr_output_file["diffuse"])
      f_albedo_out = self.transform(exr_output_file["albedo"])
      
      self.Y_diffuse = torch.div(c_diffuse_out, (f_albedo_out + epsilon))
      
        # specular
      c_specular_out = self.transform(exr_output_file["specular"])
      c_specular_out = torch.clamp(c_specular_out, min = 0.0)
      self.Y_specular = torch.log(c_specular_out+1)
      ##########
      # self.Y_specular = c_specular_out

      # X processing
      
      c_diffuse = self.transform(exr_input_file["diffuse"])
      c_diffuse_var = self.transform(exr_input_file["diffuseVariance"])
      
      c_specular = self.transform(exr_input_file["specular"])
      c_specular_var = self.transform(exr_input_file["specularVariance"])
      
      f_normal = self.transform(exr_input_file["normal"])
      self.f_albedo = self.transform(exr_input_file["albedo"])
      f_depth = self.transform(exr_input_file["depth"])
      
      max_depth = f_depth.max()
      f_depth = torch.div(f_depth,max_depth)
      
      f_normal_var = self.transform(exr_input_file["normalVariance"])
      f_albedo_var = self.transform(exr_input_file["albedoVariance"])
      f_depth_var = self.transform(exr_input_file["depthVariance"])
      
      c_diffuse_trans = torch.div(c_diffuse, (self.f_albedo + epsilon))
      c_diffuse_var_trans = torch.div(c_diffuse_var, ((self.f_albedo + epsilon)*(self.f_albedo + epsilon)).sum(dim=0).unsqueeze(dim=0))
      
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
      Gx_albedo, Gy_albedo = gradient_exr(self.f_albedo)
      Gx_depth, Gy_depth = gradient_exr(f_depth)
      
      self.X_diffuse = torch.cat((c_diffuse_trans,Gx_diffuse, Gx_albedo, Gx_depth,
                                  Gx_normal, Gy_diffuse, Gy_albedo, Gy_depth, Gy_normal,
                                  c_diffuse_var_trans, f_normal_var, f_albedo_var
                                  ,f_depth_var
                                  ),dim=0)
      
      self.X_specular = torch.cat((c_specular_trans,Gx_specular, Gx_albedo, Gx_depth,
                          Gx_normal, Gy_specular, Gy_albedo, Gy_depth, Gy_normal,
                          c_specular_var_trans, f_normal_var, f_albedo_var
                          ,f_depth_var
                          ),dim=0)
      # self.X_specular = c_specular_trans
        
    def __getitem__(self, index):
      file_n = index // (9*16)
      k = index % (9*16)
      
      i = k % 9
      j = k // 9
      
      X_diffuse = self.X_diffuse[:,file_n*720+80*i:file_n*720+80*(i+1),80*j:80*(j+1)]
      Y_diffuse = self.Y_diffuse[:,file_n*720+80*i:file_n*720+80*(i+1),80*j:80*(j+1)]
      
      X_specular = self.X_specular[:,file_n*720+80*i:file_n*720+80*(i+1),80*j:80*(j+1)]
      Y_specular = self.Y_specular[:,file_n*720+80*i:file_n*720+80*(i+1),80*j:80*(j+1)]
      
      return X_diffuse, Y_diffuse, X_specular, Y_specular
    
    def test(self):
      return self.X_diffuse, self.Y_diffuse, self.X_specular, self.Y_specular, self.f_albedo
      
    def testset(self):
      self.input_path = i_path
      self.raw_data_dir = sorted(os.listdir(path=self.input_path))
      self.preprocessing(0)
      return self.X_diffuse, self.Y_diffuse, self.X_specular, self.Y_specular, self.f_albedo
  
data = Data(transform=transform,input_path=i_path)

dataloader = DataLoader(data,batch_size=1,shuffle=True,drop_last=True)
