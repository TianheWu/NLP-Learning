import torch
import numpy as np


def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

x = torch.arange(6)
x = x.view(2, 3)
describe(x[0, :2])
# describe(torch.sum(x, dim=0))
# describe(torch.Tensor([[1, 2, 3], [4, 5, 6]]))