import torch
import matplotlib.pyplot as plt


x = torch.range(-5, 5, 0.1)
y = torch.tanh(x)
plt.plot(x.numpy(), y.numpy())
plt.show()