from preprocessing import *
import os
import random
import numpy as np
import h5py
from multiprocessing import Queue, Process, Value
import multiprocessing

# patch size = 128 in paper but, too large so 64 
class HDF5Constructor:
    def __init__(self, data_path, save_path, patch_size, num_patches, seed, train_val_ratio):
        assert sum(train_val_ratio) == 1, "Sum of train_val_ratio must be 1!"
        self.data_path = data_path
        self.save_path = save_path
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.seed = seed
        self.train_val_ratio = train_val_ratio
        
        self.exr_paths = []
        self.gt_paths = []
        self.paths = []
        
    def construct_hdf5(self):
        print("Constructing HDF5 set")
        self.get_exr_paths()
        self.get_cropped_patches()
        print("Constructing data set (hdf5) : done")
    
    def get_exr_paths(self):
        self.noisy_path = os.path.join(self.data_path, 'noisy')
        self.gt_path = os.path.join(self.data_path,"ground_truth")
        for root, dirs, files in os.walk(self.gt_path):
            print("root : ", root)
            print("dirs : ", dirs)
            files.sort()
            if not dirs:
                for file in files:
                    filename=file.split('-')[0]
                    exr_path = os.path.join(self.noisy_path, filename+"-00128spp.exr")
                    gt_path = os.path.join(self.gt_path, file)
                    if gt_path not in self.gt_paths:
                        self.gt_paths.append(gt_path)
                        self.exr_paths.append(exr_path)
                
        assert len(self.exr_paths) == len(self.gt_paths), "Samples does not equal to gts, check the data!"
        assert isinstance(self.seed, int) , "Seed must be an int"
        random.seed(self.seed)
        self.paths = [(e,g) for e,g in zip(self.exr_paths, self.gt_paths)]
        random.shuffle(self.paths)
        print("\r\t-Get exr paths : done",end='')
        print(self.paths)
        
    def worker(self, queues, path_mapping, name_shape_mapping, lock, v):
        while not queues[0].empty() or not queues[1].empty():
            v.value += 1
            print("\r\t-Generating patches: %d / %d" % (v.value, len(self.paths)-3), end='')
            if not queues[0].empty():
                path = queues[0].get()
                dataset = 'train'
            elif not queues[1].empty():
                path = queues[1].get()
                dataset = 'val'
            cropped, patches = get_cropped_patches(path[0], path[1], self.patch_size, self.num_patches)

            lock.acquire()
            with h5py.File(path_mapping[dataset], 'a') as hf:
                for key in name_shape_mapping.keys():
                    temp = [c[key] for c in cropped]
                    hf[key].resize((hf[key].shape[0] + len(temp)), axis=0)
                    hf[key][-len(temp):] = temp
            lock.release()


    def get_cropped_patches(self):
        patch_count = 0
        train_save_path = os.path.join(self.save_path, "train.h5")
        val_save_path = os.path.join(self.save_path, "val.h5")
        path_mapping = {'train': train_save_path, 'val': val_save_path}
        name_shape_mapping = {'noisy': (None, 80, 80, 3),
                              'gt': (None, 80, 80, 3),
                              'aux': (None, 80, 80, 7)}
        queues = [Queue() for i in range(2)]  # [train_queue, val_queue]
        # the first 2 paths are used to initiate h5py files
        for i in range(2, len(self.paths)):
            if (i - 2) < int(self.train_val_ratio[0] * (len(self.paths) - 2)):
                queues[0].put(self.paths[i])
            else:
                queues[1].put(self.paths[i])
        # initiate h5py files
        print("\n\r\t-Initiating h5py files", end='')
        for i, n in enumerate(['train', 'val']):
            with h5py.File(path_mapping[n], 'w') as hf:
                cropped, patches = get_cropped_patches(self.paths[i][0], self.paths[i][1], self.patch_size,
                                                       self.num_patches)
                patch_count += len(cropped)
                for key in name_shape_mapping.keys():
                    temp = np.array([c[key] for c in cropped])
                    hf.create_dataset(key, data=temp, maxshape=name_shape_mapping[key], compression="gzip",
                                      chunks=True)
                    print(np.shape(temp))
        print("\r\t-Initiating h5py files: done")

        # start processes
        lock = multiprocessing.Lock()  # to ensure only one process writes to the file at once
        done_exr_count = Value("i", 0)
        pool = [Process(target=self.worker, args=(queues, path_mapping, name_shape_mapping, lock, done_exr_count))
                for i in range(multiprocessing.cpu_count() - 1)]
        for p in pool:
            p.start()
        for p in pool:
            p.join()

        print("\r\t-Generating patches: done")

# n = NumpyConstructor("/rec/pvr1/2021_Summer_Internship/raw_data/train_set","",64,7,0,0)
# n.get_exr_paths()