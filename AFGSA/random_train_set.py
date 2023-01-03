import os
import random
from shutil import copyfile

input_path = "/rec/pvr1/deep_learning_denoising/renderings/"
output_path = "/rec/pvr1/2021_Summer_Internship/raw_data/"
num = 5

dir=sorted(os.listdir(path=input_path))
for e in dir:
    if e == "thumbs":
        continue
    sub_dir = sorted(os.listdir(input_path + e))
    sub_len = len(sub_dir)//5
    ran_test = random.sample(range(sub_len),80)
    for ran in ran_test:
        copyfile("{}{}/{}".format(input_path,e,sub_dir[ran*5]),"{}train_set/noisy/{}".format(output_path,sub_dir[ran*5]))
        copyfile("{}{}/{}".format(input_path,e,sub_dir[ran*5+4]),"{}train_set/ground_truth/{}".format(output_path,sub_dir[ran*5+4]))
    