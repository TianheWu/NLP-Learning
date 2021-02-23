import torch


x = [1, 2, 3, 4, 5]
x = torch.LongTensor(x)
print(x)
print(x.unsqueeze(0))
print(x.unsqueeze(1))
print(x.unsqueeze(1).unsqueeze(0))