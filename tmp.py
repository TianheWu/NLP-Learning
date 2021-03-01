import torch
import numpy as np
#from TextCNN.textCNN import TCNN
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from pandas import read_csv, DataFrame
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os


def get_data():
    f = open(r'/home/wutianhe/NLP/IMDB Dataset.csv', encoding='UTF-8')
    data = read_csv(f, names=np.arange(2))
    data_df = DataFrame(data)

    x = data_df.iloc[1:40000, [0]]
    y = data_df.iloc[1:40000, [1]]

    sentences = []
    labels = []

    for i in range(len(x)):
        sen = []
        s = x.iloc[i, 0]
        s = s.replace('.', ' ')
        s = s.replace('!', ' ')
        s = s.replace('-', ' ')
        s = s.replace('/', ' ')
        s = s.replace('_', ' ')
        s = s.replace('#', ' ')
        s = s.replace('$', ' ')
        s = s.replace(';', ' ')
        s = s.replace('[', ' ')
        s = s.replace(']', ' ')
        s = s.replace('~', ' ')
        s = s.replace('|', ' ')
        s = s.replace('(', ' ')
        s = s.replace(')', ' ')
        s = s.replace('=', ' ')
        s = s.replace(',', ' ')
        for word in s.split():
            sen.append(word)
        sen = ' '.join(sen)
        sentences.append(sen)

    for i in range(len(y)):
        if y.iloc[i, 0] == 'positive':
            labels.append(1)
        else:
            labels.append(0)

    return sentences, labels


sentences, labels = get_data()

embedding_size = 5
sequence_length = 200
num_classes = len(set(labels))
batch_size = 100


# build dictionary between word and seq
word_list = ' '.join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)


def make_data(sentences, labels):
    inputs = []
    i = 0
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])
        if len(inputs[i]) >= 200:
            inputs[i] = inputs[i][:200]
        else:
            ext = [0 for i in range(200 - len(inputs[i]))]
            inputs[i] = inputs[i] + ext
        i += 1


    targets = []
    for out in labels:
        targets.append(out)

    return inputs, targets


input_batch, target_batch = make_data(sentences, labels)

train_x = input_batch[0:35000]
train_y = target_batch[0:35000]
test_x = input_batch[35000:40000]
test_y = target_batch[35000:40000]



print("XXXXXXXX")
model = TCNN()
model.load_state_dict(torch.load('tcnn.pt'))
model.eval()
pre_y = model(test_x)
print(pre_y)