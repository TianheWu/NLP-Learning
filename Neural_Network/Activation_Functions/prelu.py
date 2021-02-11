import torch
import matplotlib.pyplot as plt


prelu = torch.nn.PReLU(num_parameters=1)
x = torch.range(-5., 5., 0.1)
y = prelu(x)
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.show()