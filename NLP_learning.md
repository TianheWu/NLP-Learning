## NLP 学习笔记

#### 学习资源：https://www.bookstack.cn/read/nlp-pytorch-zh/136a55e29241a22f.md



### 2021-1-23 chapter 1

#### 几个常见名词

**监督学习**：使用标记的训练示例进行学习

**观察**：输入（我们想要预测的东西），用x表示

**目标**：被语言的事情，用y表示（被称为ground truth）

**Model**：数学表达式或函数，接受一个观察值x，预测目标标签的值

**Parameters**：权重，w

**Predictions**：预测，模型在给定观测值的情况下所猜测目标的值

**Loss function**：损失函数，比较预测与训练数据中观测目标之间的距离函数

{time, fruit, flies, like, a, an, arrow, banana}



### 2021-1-26 chapter 1

**PS：网上代码写的有问题，我这里进行了修改：**

```python
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

vocab = ['an', 'arrow', 'banana', 'flies', 'fruit', 'like', 'time']
corpus = ['Time flies flies like an arrow.',
          'Fruit flies like a banana.']

one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(one_hot, annot=True,
            cbar=False, xticklabels=vocab,
            yticklabels=['Sentence 1', 'Sentence 2'])
plt.show()
```

**TF-IDF = TF(w) * IDF(w)**

TF表示对更频繁的单词进行加权，IDF表示惩罚常见的符号，并奖励向量表示中的罕见符号

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

vocab = ['an', 'arrow', 'banana', 'flies', 'fruit', 'like', 'time']
corpus = ['Time flies flies like an arrow.',
          'Fruit flies like a banana.']

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(tfidf, annot=True, cbar=False, xticklabels=vocab,
            yticklabels= ['Sentence 1', 'Sentence 2'])

plt.show()
```



### 2021-1-27 chapter 1

在安装了3天多pytorch后，终于安装好了。就别指望在windows下安装了，我是在linux ubuntu环境下安装的，要先安装好**Anaconda**，pycharm上的解释器要配置成Anaconda，再安装conda包管理器，设置python环境，最后添加清华的镜像，才能快速下载，不然下载速度就和龟速一样。

其中conda下载+配置网址为：https://blog.csdn.net/weixin_43840215/article/details/89599559

加速下载pytorch网址为：https://cloud.tencent.com/developer/article/1588508

最后导入pycharm中：https://zhuanlan.zhihu.com/p/263493426

创建张量的小代码以及一些函数等：

```python
import torch


def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))


describe(torch.Tensor(2, 3))

# torch.rand(2, 3)  随机初始化值区间上的均匀分部(0, 1)
# torch.randn(2, 3) 正态分布
# torch.zeros(2, 3) 0张量
# torch.ones(2, 3)  1张量

# 可以用x.fill()来填充特定的值

# 可以通过列表来初始化，但是要注意Tensor()里面只能是一个列表：
# describe(torch.Tensor([[1, 2, 3], [4, 5, 6]]))

# 可以用Numpy数组初始化
# npy = np.random.rand(2, 3)
# describe(torch.from_numpy(npy))

# 类型转换
# x = torch.FloatTensor()
# x = x.long(), x = x.float()

# 普通操作
# +，-，*，/   .add(x, x)

# 维度操作
# x = torch.arange(6)
# x = x.view(2, 3)
# describe(torch.sum(x, dim=0))  行表示0，列表示1，dim=0表示所有行相加，dim=1表示所有列相加
# torch.sum()后会降维
# torch.transpose(x, 0, 1) 转置操作
```

访问GPU时，要调用**CUDA API**。要对CUDA和非CUDA对象进行操作，我们需要确保它们在同一设备上。

```python
import torch
print (torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # .device("cpu")
print(device)

x = torch.rand(3, 3).to(device) # x = x.to(<device>)
describe(x)
```

**将数据从GPU来回移动是非常昂贵的！！！！！！！！！！**



### 2021-1-28 chapter 2

语料库包含**原始文本**和**元数据**。

**原始文本**：字符(字节)序列，但是大多数时候将字符分组成连续的称为令牌(Tokens)的连续单元是有用的。在英语中，令牌(Tokens)对应由空格字符或标点分隔的单词和数字序列。

**元数据**：是与文本相关联的任何辅助信息，例如标识符，标签和时间戳。

**PS：这里我要插一句：要安装spacy.load('en')这里要安装pycharm中的en包，不要从网上下载，直接在pycharm中搜索就能找到！！！！！！！！！！！！！！！！！**

根据不同的文本处理包进行**令牌化Tokenizing：**

```python
import spacy
nlp = spacy.load(‘en’)
text = “Mary, don’t slap the green witch”
print([str(token) for token in nlp(text.lower())])

# Output:
# ['mary', ',', 'do', "n't", 'slap', 'the', 'green', 'witch', '.']

from nltk.tokenize import TweetTokenizer
tweet = u"Snow White and the Seven Degrees
    #MakeAMovieCold@midnight:-)"
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet.lower()))

# Output:
# ['snow', 'white', 'and', 'the', 'seven', 'degrees', '#makeamoviecold', '@midnight', ':-)']
```

**ngram**是文本中出现的固定长度(n)的连续令牌序列。

```python
def n_grams(text, n):
    '''
    takes tokens or text, returns a list of n grams
    '''
    return [text[i:i+n] for i in range(len(text)-n+1)]
cleaned = ['mary', ',', "n't", 'slap', green', 'witch', '.']
print(n_grams(cleaned, 3))
           
# Output:
# [['mary', ',', "n't"],
# [',', "n't", 'slap'],
# ["n't", 'slap', 'green'],
# ['slap', 'green', 'witch'],
# ['green', 'witch', '.']]
```

**Lemmas**是单词的词根形式。考虑动词fly。它可以被屈折成许多不同的单词——flow、fly、flies、flying、flow等等——而fly是所有这些看似不同的单词的Lemmas。

**Lemmatization:**

```python
import spacy
nlp = spacy.load(‘en’)
doc = nlp(u"he was running late")
for token in doc:
    print('{} --> {}'.format(token, token.lemma_))

# Output:
# he --> he
# was --> be
# running --> run
# late --> late
```

词性标注**POS Tagging**：

```python
import spacy
nlp = spacy.load(‘en’)
doc = nlp(u"Mary slapped the green witch.")
for token in doc:
    print('{} - {}'.format(token, token.pos_))
```

浅解析(Shallow parsing)：推导出由名词、动词、形容词等语法原子组成的高阶单位。

```python
import spacy
nlp = spacy.load(‘en’)
doc  = nlp(u"Mary slapped the green witch.")
for chunk in doc.noun_chunks:
    print '{} - {}'.format(chunk, chunk.label_)

# Output:
# Mary - NP
# the green witch - NP
```

浅层解析识别短语单位，而识别它们之间关系的任务称为解析(parsing)。



1-29和女朋友出去玩了，没有进行学习



### 2021-1-30 chapter 3 神经网络的基本组件

#### 模型：感知器

输入x（是向量：因为不仅有一个输入），输出y，权重w（同x），偏量b，激活函数f。权重和偏量都从数据学习激活函数是精心挑选的取决于网络的网络设计师的直觉和目标输出。

#### 激活函数

sigmoid，tanh，relu，prelu，softmax

#### 损失函数

**Mean Squared Error Loss**：MSE就是预测值与目标值之差的平方的平均值。

**Categorical Cross-Entropy Loss**：通常用于多类分类设置。

**Binary Cross-Entropy**：有时，我们的任务包括区分两个类——也称为二元分类。在这种情况下，利用二元交叉熵损失是有效的。

#### 优化器

当模型产生预测，损失函数测量预测和目标之间的误差时，优化器使用错误信号更新模型的权重。最简单的形式是，有一个超参数控制优化器的更新行为。这个超参数称为**学习率**，它控制错误信号对更新权重的影响。

#### 神经网络模型 Gradient-Based Supervised Learning

将所有合起来后，能够得到一个简单的神经网络模型**Gradient-Based Supervised Learning**，它的算法为：

1. 使用名为zero_grad()的函数**清除**当前存储在模型(感知器)对象中的所有记帐信息，例如梯度。
2. 模型**计算**给定输入数据(x_data)的输出(y_pred)。
3. **比较**模型输出(y_pred)和预期目标(y_target)来计算损失。这正是有监督训练信号的有监督部分。
4. PyTorch损失对象(criteria)具有一个名为bcakward()的函数，该函数迭代地通过**计算图向后传播损失**，并将其梯度**通知**每个参数。
5. 优化器(opt)用一个名为step()的函数指示参数如何在知道梯度的情况下**更新**它们的值。

现在我才知道，在做keystone的时候，我将大数据变为小数据分批处理的方法术语叫**batch**：整个训练数据集被划分成多个批(batch)。

在文献和本书中，术语minibatch也可以互换使用，而不是“batch”来强调每个**batch都明显小于训练数据的大小**;例如，训练数据可能有数百万个，而小批数据可能只有几百个。梯度步骤的每一次迭代都在一批数据上执行。名为**batch_size**的超参数指定批次的大小。在多个批处理(通常是有限大小数据集中的批处理数量)之后，训练循环完成了一个**epoch**。epoch是一个完整的训练迭代。如果每个epoch的批数量与数据集中的批数量相同，那么epoch就是对数据集的完整迭代。

**A supervised training loop for a Perceptron and binary classification**

```python
# each epoch is a complete pass over the training data
for epoch_i in range(n_epochs):
    # the inner loop is over the batches in the dataset
    for batch_i in range(n_batches):
        # Step 0: Get the data
        x_data, y_target = get_toy_data(batch_size)
        # Step 1: Clear the gradients
        perceptron.zero_grad()
        # Step 2: Compute the forward pass of the model
        y_pred = perceptron(x_data, apply_sigmoid=True)
        # Step 3: Compute the loss value that we wish to optimize
        loss = bce_loss(y_pred, y_target)
        # Step 4: Propagate the loss signal backward
        loss.backward()
        # Step 5: Trigger the optimizer to perform one update
        optimizer.step()
```

标准实践是将数据集分割为三个随机采样的分区，称为训练、验证和测试数据集，或者进行k-fold交叉验证。分成三个分区是两种方法中比较简单的一种，因为它只需要一次计算。您应该采取预防措施，确保在三个分支之间的类分布保持相同。一个常见的分割百分比是预留70%用于培训，15%用于验证，15%用于测试。



### 2021-1-31 chapter 3

之前的例子训练了固定次数的模型。虽然这是最简单的方法，但它是任意的和不必要的。正确度量模型性能的一个关键功能是使用该度量来知道何时应该停止训练。

#### 启发式方法 ------ 早期停止

通过跟踪验证数据集上从一个epoch到另一个epoch的性能并注意性能何时不再改进来的工作。如果业绩继续没有改善，训练将终止。在结束训练之前需要等待的时间称为**耐心**。一般来说，模型停止改进某些数据集的时间点称为**模型收敛的时间点**。在实际应用中，我们很少等待模型完全收敛，因为收敛是耗时的，而且**会导致过拟合**。

#### 正则化

**L2正则化**：平滑约束。在PyTorch中，可以通过在**优化器**中设置weight_decay参数来控制这一点。weight_decay值越大，优化器选择的解释就越流畅;也就是说，L2正则化越强。

**dropout**

### 2021-2-01 chapter 4

#### 多层感知器(MLP)

感知器将数据向量作为输入，计算出一个输出值。在MLP中，许多感知器被分组，以便单个层的输出是一个新的向量，而不是单个输出值。

#### 卷积神经网络(CNN)

翻遍了全网，这个是讲的最清晰的

https://blog.csdn.net/Tink1995/article/details/104528624

#### 暂存问题：

