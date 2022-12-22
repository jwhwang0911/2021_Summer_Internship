import torch
import torch.nn.functional as F
import exr
import torch.nn as nn
import math
from PIL import Image


def gradient_exr(x):
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)
    x = x.unsqueeze(0)

    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dx, dy = right - left, bottom - top 
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    
    dx = dx.squeeze(dim=0)
    dy = dy.squeeze(dim=0)
    
    # # If you want to change to numpy then use it
    # dx = dx.data.numpy().swapaxes(1,2).swapaxes(0,2) 
    # dy = dy.data.numpy().swapaxes(1,2).swapaxes(0,2) 
    

    return dx, dy