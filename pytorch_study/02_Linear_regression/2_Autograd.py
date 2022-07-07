import torch

if __name__ == "__main__":
    w = torch.tensor(2.0, requires_grad=True)
    
    y = w**2
    z = 2*y + 5
    
    y.backward()
    print('수식을 w로 미분한 값 : ',(w.grad))