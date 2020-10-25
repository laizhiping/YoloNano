import torch
b = torch.Tensor([[1,2,3],[4,5,6]])
print(b)
index_1 = torch.LongTensor([[0,1],[2,0]])
index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
print(torch.gather(b, dim=1, index=index_1))
print(torch.gather(b, dim=0, index=index_2))