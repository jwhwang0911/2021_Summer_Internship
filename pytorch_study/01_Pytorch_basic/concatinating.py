import torch

def concatenate() :
    x = torch.FloatTensor([[1, 2], [3, 4]])
    y = torch.FloatTensor([[5, 6], [7, 8]])
    print(torch.cat([x,y],dim=0))
    print(torch.cat([x,y],dim=1))

def stacking():
    x = torch.FloatTensor([1, 4])
    y = torch.FloatTensor([2, 5])
    z = torch.FloatTensor([3, 6])
    
    print(torch.stack([x, y, z], dim=1))
    # tensor([[1., 2., 3.],[4., 5., 6.]])
    
def zero_ones():
    x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]]) # 2*3 size tensor
    print(x)
    print(torch.zeros_like(x))
    print(torch.ones_like(x))
    
def in_place_opereation():
    x = torch.FloatTensor([[1, 2], [3, 4]])
    print(x.mul_(2.))
    print(x)
    

if __name__ == "__main__":
    print('concatination')
    concatenate()
    print('stacking')
    stacking()
    print('zero_like, one_like')
    zero_ones()
    print('inplace operations')
    in_place_opereation()