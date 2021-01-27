## NLP 学习笔记

#### 学习资源：https://www.bookstack.cn/read/nlp-pytorch-zh/136a55e29241a22f.md



### 2021-1-23 chapter 1

监督学习：使用标记的训练示例进行学习

观察：输入（我们想要预测的东西），用x表示

目标：被语言的事情，用y表示（被称为ground truth）

Model：数学表达式或函数，接受一个观察值x，预测目标标签的值

Parameters：权重，w

Predictions：预测，模型在给定观测值的情况下所猜测目标的值

Loss function：损失函数，比较预测与训练数据中观测目标之间的距离函数

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

访问GPU时，要调用CUDA API。要对CUDA和非CUDA对象进行操作，我们需要确保它们在同一设备上。

```python
import torch
print (torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # .device("cpu")
print(device)

x = torch.rand(3, 3).to(device) # x = x.to(<device>)
describe(x)
```

**将数据从GPU来回移动是非常昂贵的！！！！！！！！！！**

