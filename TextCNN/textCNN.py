import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

sentences = ['i love you', 'he loves me', 'she likes baseball', 'i hate you', 'sorry for that', 'this is awful']
labels = [1, 1, 1, 0, 0, 0]

embedding_size = 2
sequence_length = len(sentences[0])
num_classes = len(set(labels))
batch_size = 3

# build dictionary between word and seq
word_list = ' '.join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)


def make_data(sentences, labels):
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])

    targets = []
    for out in labels:
        targets.append(out)

    return inputs, targets


input_batch, target_batch = make_data(sentences, labels)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, shuffle=True)


class TCNN(nn.Module):

    def __init__(self):
        super(TCNN, self).__init__()
        self.W = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_channel,
                kernel_size=(2, embedding_size),
                stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.out = nn.Linear(output_channel, num_classes)

    def forward(self, X):

        batch_size = X.shape[0]
        embedding_X = self.W(X)
        embedding_X = embedding_X.unsqueeze(1)
        conved = self.conv(embedding_X)
        flatten = conved.view(batch_size, -1)
        output = self.out(flatten)
        return output

model = TCNN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = loss_func(pred, batch_y)
        if epoch % 1000 == 0:
            print('EPOCH: ', epoch, '| train loss: %.4f' % loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_text = 'i likes you'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests)

model = model.eval()
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text, 'is bad mean...')
else:
    print(test_text, 'is good mean...')
