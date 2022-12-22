import exr
import os

exec(open("diffuse.py").read())
exec(open("specular.py").read())

for i in range(20):
    dif_temp_dir = sorted(os.listdir("/rec/pvr1/deep_learning_denoising/KPCN_new/test/diffuse/"+str(i*10)))
    spe_temp_dir = sorted(os.listdir("/rec/pvr1/deep_learning_denoising/KPCN_new/test/specular/"+str(i*10)))
    
    for idx in range(4):
        dif = exr.read("/rec/pvr1/deep_learning_denoising/KPCN_new/test/diffuse/"+str(i*10)+"/"+dif_temp_dir[idx])
        spe = exr.read("/rec/pvr1/deep_learning_denoising/KPCN_new/test/specular/"+str(i*10)+"/"+spe_temp_dir[idx])
        exr.write("/rec/pvr1/deep_learning_denoising/KPCN_new/test/estimate/"+str(i*10)+"/estimate_"+str(idx+1)+".exr",dif+spe)

dif = exr.read("/rec/pvr1/deep_learning_denoising/KPCN_new/test/diffuse.exr")
spe = exr.read("/rec/pvr1/deep_learning_denoising/KPCN_new/test/specular.exr")
exr.write("/rec/pvr1/deep_learning_denoising/KPCN_new/test/estimate.exr",dif+spe)