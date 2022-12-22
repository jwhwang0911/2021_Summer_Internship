import exr
import os

exec(open("diffuse.py").read())
exec(open("specular.py").read())

dif_dir = sorted(os.listdir("/rec/pvr1/deep_learning_denoising/KPCN/KPCN_test/diffuse/"))
spe_dir = sorted(os.listdir("/rec/pvr1/deep_learning_denoising/KPCN/KPCN_test/specular/"))

for i in range(len(dif_dir)):
    dif = exr.read("/rec/pvr1/deep_learning_denoising/KPCN/KPCN_test/diffuse/"+dif_dir[i])
    spe = exr.read("/rec/pvr1/deep_learning_denoising/KPCN/KPCN_test/specular/"+spe_dir[i])
    
    exr.write("/rec/pvr1/deep_learning_denoising/KPCN/KPCN_test/c_estimate/estimate_"+str(i)+".exr",dif+spe)


dif = exr.read("/rec/pvr1/deep_learning_denoising/KPCN/KPCN_test/diffuse_test.exr")
spe = exr.read("/rec/pvr1/deep_learning_denoising/KPCN/KPCN_test/specular_test.exr")

exr.write("/rec/pvr1/deep_learning_denoising/KPCN/KPCN_test/c_estimate.exr",dif+spe)