## NLP 学习笔记

#### 学习资源：https://www.bookstack.cn/read/nlp-pytorch-zh/136a55e29241a22f.md

### 2021-1-23 chapter 1

**监督学习：**使用标记的训练示例进行学习

**观察：**输入（我们想要预测的东西），用x表示

**目标：**被语言的事情，用y表示（被称为ground truth）

**Model：**数学表达式或函数，接受一个观察值x，预测目标标签的值

**Parameters：**权重，w

**Predictions：**预测，模型在给定观测值的情况下所猜测目标的值

**Loss function：**损失函数，比较预测与训练数据中观测目标之间的距离函数

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

在安装了3天多pytorch后，终于安装好了。就别指望在windows下安装了，我是在linux ubuntu环境下安装的，要先安装好**Anaconda**，再安装conda包管理器，设置python环境，最后添加清华的镜像，才能快速下载，不然下载速度就和龟速一样。

其中conda下载+配置网址为：https://blog.csdn.net/weixin_43840215/article/details/89599559

加速下载pytorch网址为：https://cloud.tencent.com/developer/article/1588508

最后导入pycharm中：https://zhuanlan.zhihu.com/p/263493426



