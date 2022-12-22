import os
import exr

exec(open("iter_diffuse.py").read())
exec(open("iter_specular.py").read())

dif_dir = sorted(os.listdir("/rec/pvr1/deep_learning_denoising/DPCN/DPCN_test/iter_data/diffuse/"))
spe_dir = sorted(os.listdir("/rec/pvr1/deep_learning_denoising/DPCN/DPCN_test/iter_data/specular/"))

for i in range(len(dif_dir)):
    dif = exr.read("/rec/pvr1/deep_learning_denoising/DPCN/DPCN_test/iter_data/diffuse/"+dif_dir[0])
    spe = exr.read("/rec/pvr1/deep_learning_denoising/DPCN/DPCN_test/iter_data/specular/"+spe_dir[0])
    
    exr.write("/rec/pvr1/deep_learning_denoising/DPCN/DPCN_test/iter_data/c_estimate/estimate_"+str(i)+".exr",dif+spe)
    
dif = exr.read("/rec/pvr1/deep_learning_denoising/DPCN/DPCN_test/iter_data/diffuse_test.exr")
spe = exr.read("/rec/pvr1/deep_learning_denoising/DPCN/DPCN_test/iter_data/specular_test.exr")

exr.write("/rec/pvr1/deep_learning_denoising/DPCN/DPCN_test/iter_data/c_estimate.exr",dif+spe)