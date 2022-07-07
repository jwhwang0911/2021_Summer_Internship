import torch

# .matmul <-- Matrix multiplication
# .mul or * <-- element multiplication, A.mul(B) == sigma ( A_(i,j) * B_(i,j) )

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])

# m1 -> 2*2 matirx, m2 -> 2*1 matrix // so m1 * m2 = 2*1 matrix
print(m1.matmul(m2))

print(m1)
# m1 is same after the mathmetical multiplication

# m2 broadcast as [[1,1],[2,2]]
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2)) # [[1,2],[6,8]]

#mean with no parameter
t = torch.FloatTensor([1,2])
print(t.mean())
# 1.5
t = torch.FloatTensor([[1,2],[3,4]])
print(t.mean())
#2.5

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.sum()) # 단순히 원소 전체의 덧셈을 수행
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거


#mean with parameter

