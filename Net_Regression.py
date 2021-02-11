import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x ** 2 + 0.2 * torch.rand(x.size())

x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        hid_x = F.relu(self.hidden(x))
        out_x = self.predict(hid_x)
        return out_x


net = Net(1, 10, 1)
print(net)
plt.ion()

optimizer = torch.optim.SGD(net.parameters(), lr=0.3)
loss_func = torch.nn.MSELoss()

for ep in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if ep % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.pause(0.1)

plt.ioff()
plt.show()


