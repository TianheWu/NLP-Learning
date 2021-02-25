import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os


def get_data(path, binary):
    files = os.listdir(path)
    sentences = []
    new_sentences = []
    labels = [binary for i in range(700)]

    for file in files:
        position = path + '/' + file
        with open(position, 'r', encoding='unicode_escape') as f:
            data = f.read()
            sentences.append(data)

    for s in sentences:
        s = s.replace('\n', ' ')
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
        new_sentences.append(s[0:300])

    return new_sentences, labels


path_pos = "/home/wutianhe/NLP/data/mix20_rand700_tokens_cleaned/tokens/pos"
path_neg = "/home/wutianhe/NLP/data/mix20_rand700_tokens_cleaned/tokens/neg"

sentences_pos, labels_pos = get_data(path_pos, 1)
sentences_neg, labels_neg = get_data(path_neg, 0)

sentences = sentences_pos + sentences_neg
labels = labels_pos + labels_neg

embedding_size = 100
sequence_length = len(sentences[0])
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
        if len(inputs[i]) >= 50:
            inputs[i] = inputs[i][:50]
        else:
            ext = [0 for i in range(50 - len(inputs[i]))]
            inputs[i] = inputs[i] + ext
        i += 1

    targets = []
    for out in labels:
        targets.append(out)

    return inputs, targets


input_batch, target_batch = make_data(sentences, labels)
print(input_batch)

train_x = input_batch[:980]
train_y = target_batch[:980]
test_x = input_batch[980:]
test_y = target_batch[980:]
train_x, train_y = torch.LongTensor(train_x), torch.LongTensor(train_y)
test_x, test_y = torch.LongTensor(test_x), torch.LongTensor(test_y)

train_ds = Data.TensorDataset(train_x, train_y)
train_dl = Data.DataLoader(train_ds, batch_size, True)
test_ds = Data.TensorDataset(test_x, test_y)
test_dl = Data.DataLoader(test_ds, batch_size, False)
