import torch
import torch.nn.functional as F


# Low level
# ___________________________________________________________________
torch.manual_seed(1)

z = torch.FloatTensor([1,2,3]) # Softmax input

hypothesis = F.softmax(z,dim=0)
print(hypothesis)

print(hypothesis.sum()) # check sum is 1.0



z = torch.rand(3, 5,requires_grad=True) # random 3*5 matrix input of softmax

hypothesis = F.softmax(z,dim=1)

print(hypothesis)
print(hypothesis.sum(dim=1))

y = torch.randint(5, (3,)).long()
print(y)
print(y.unsqueeze(1))

# 모든 원소가 0의 값을 가진 3 × 5 텐서 생성
y_one_hot = torch.zeros_like(hypothesis) 
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)


#high level
# ___________________________________________________________________

# High level
F.log_softmax(z, dim=1) # which is same as
torch.log(F.softmax(z, dim=1))


# Cost function
# Low level
# 첫번째 수식
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

# 두번째 수식
(y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()

# High level | Negative Log Likelihood = nll
# 세번째 수식
F.nll_loss(F.log_softmax(z, dim=1), y)

# 네번째 수식
F.cross_entropy(z, y)