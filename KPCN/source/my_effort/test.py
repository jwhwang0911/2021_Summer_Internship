import torch
import torch.nn.functional as F

a = torch.FloatTensor([[[
    [1.,2.,3.,4.,5.],
    [6.,7.,8.,9.,10.],
    [11.,12.,13.,14.,15.],
    [16.,17.,18.,19.,20.]
],[
    [1.,2.,3.,4.,5.],
    [6.,7.,8.,9.,10.],
    [11.,12.,13.,14.,15.],
    [16.,17.,18.,19.,20.]
],[
    [1.,2.,3.,4.,5.],
    [6.,7.,8.,9.,10.],
    [11.,12.,13.,14.,15.],
    [16.,17.,18.,19.,20.]
]]])

kernel = torch.FloatTensor([[[
    [1.,2.,3.],
    [4.,5.,6.],
    [7.,8.,9.]
]]])

def apply_kernel(kernel:torch.Tensor, target:torch.Tensor, batch:int, height:int, length:int)->torch.Tensor: # kernel : batchsize * 441 * 80 * 80 , target : batchsize * 3 * 100 * 100
    #kernel = torch.ones(batch,441,height,length).cuda()
    result = torch.clone(target)
    j = 0
    for i in range(batch):
        for x in range(height):
            for y in range(length):
                # subkernel = kernel[i,:,x,y].view(1,1,3,3)
                pixel = target[i:i+1,:,x:x+3,y:y+3]
                result= (pixel * kernel)
                print(result)

k = torch.nn.ZeroPad2d(3)
a = k(a)

apply_kernel(kernel=kernel, target = a, batch=1, height=4, length=5 )