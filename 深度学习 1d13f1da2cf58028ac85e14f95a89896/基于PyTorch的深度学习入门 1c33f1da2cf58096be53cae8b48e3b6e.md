# 基于PyTorch的深度学习入门

日期: 03/27/2025
状态: 进行中

<aside>
⌨️

**写在前面**
常用的第三方库包括：PyTorch Pandas NumPy Scikit-learn等

我们用代码实战的方式初步入门

参考文章：

[深度学习入门模型建立](https://bohrium.dp.tech/notebooks/4914094074?utm_source=bilibili001)

资源：
[深入浅出PyTorch](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/index.html)

</aside>

# 一、**PyTorch 深度学习模型的建立范式**

模型建立范式中的五个步骤如下：

- **准备数据**
- **定义模型**
- **训练模型**
- **评估模型**
- **做出预测**

---

## 1.准备数据

### 1.1 数据导入

常用Pandas来加载，初步处理数据，常见数据格式为.csv .xlsx .json等

- 例如，我们通常用`pd.read_csv(path)` 以路径的方式读取csv，得到一个pandas.DataFrame类型的数据表格，其拥有可选的行列标签，灵活的数据类型等功能特点

> **CSV（Comma-Separated Values）** 
.csv文件是一种简单的文本文件格式，用于存储**表格数据**（例如数据库、电子表格中的数据）。CSV 文件通过逗号（`,`）来分隔每一列的数据，通常每行代表一条记录（或数据条目），适合用来存储**结构化数据**
本地路径的读取速度相对更快（程序运行慢可能是读取在线数据集的网络访问出问题）
> 

---

### 1.2 数据封装

对于这个初始的数据，我们通常需要做进一步处理，如通过继承pytorch的Dataset来封装数据，得到一个pytorch风格的数据集合。首先需要提取数据中的输入输出属性，这时常用数组切片,如：

```python
class CSVDataset(Dataset):
	def __init__(self, path):
		df = pd.read_csv(path, header=None)
		
		# 提取输入属性，.values[:, :-1]中，: 代表提取所有行， :-1代表提取从第一列到倒数第二列
		self.X = df.values[:, :-1]
		
		# 提取输出，所有行和最后一列
		self.y = df.values[:, -1]
		
#多维数组的切片方法同理
```

> **`header=None` 的作用：**
> 
> - 当设置 `header=None` 时，Pandas 会将 CSV 文件的第一行数据（通常是列名）当作普通数据处理，而不是作为列名。此时，所有的数据都会被加载到 DataFrame 中，并且默认会自动为每一列分配整数型的列索引（从 0 开始）

> pandas.DataFrame的.values属性返回的是一个**NumPy数组**，因为pandas本身是基于NumPy实现的，DataFrame中存储的数据默认是NumPy数组类型。
> 

数据的输出属性通常不是数，如各种分类任务的结果可能为字符串，但是神经网络或其他机器学习算法通常要求目标变量（`y`）是数值型的，所以就需要对输出做一个 **字符—数值** 的映射，例如，如果你有一个分类任务，类别标签可能是 `"cat"`, `"dog"`, 和 `"bird"`，`LabelEncoder` 会将它们转换为数字标签，如 `0`, `1`, 和 `2`

> `LabelEncoder().fit_transform()`
从sklearn库import，用于将类别标签转换为整数映射，如果将类实例化，则可以查看这个映射关系，如`encoder.classes_`，存储的可能是`['cat''dog''bird']`
> 

经过数据输入与处理后，我们搭建了基本的，符合pytorch生态的数据框架

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder #用于输出的字-值映射
from torch.utils.data import Dataset #基本数据集结构
from torch.utils.data import random_split #根据尺寸随机划分训练集与测试集

class CSVDataset(Dataset):
  def __init__(self, path):
      df = pd.read_csv(path, header=None)
      self.X = df.values[:, :-1]  # 提取输入属性
      self.y = df.values[:, -1]  # 提取输出属性
      self.X = self.X.astype('float32') # 转换数据类型为32位浮点型
      self.y = LabelEncoder().fit_transform(self.y) # 使用浮点型标签编码原输出

  # 定义获得数据集长度的方法
  def __len__(self):
      return len(self.X)

  # 定义获得某一行数据的方法，得到是一个pair，不是一维数组
  def __getitem__(self, idx):
      return [self.X[idx], self.y[idx]]
  
  # 在类内部定义划分训练集和测试集的方法，在本例中，训练集比例为 0.67，测试集比例为 0.33
  def get_splits(self, n_test=0.33):
      test_size = round(n_test * len(self.X))
      train_size = len(self.X) - test_size
      return random_split(self, [train_size, test_size]) # 根据尺寸划分训练集和测试集
```

- 基本交互

```python
data_path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'

dataset = CSVDataset(data_path)

print(f'输入矩阵的形状是：{dataset.X.shape}')

print(f'输出矩阵的形状是：{dataset.y.shape}')

# len() 方法本质上是调用类内部的 __len__() 方法，所以以下方法是等效的。
print(len(dataset))
print(dataset.__len__())

# dataset[] 方法本质上是调用类内部的 __getitem__ 方法，所以以下方法是等效的。
print(dataset[149])
print(dataset.__getitem__(149))
```

- 得到输出
    
    ![Snipaste_2025-03-27_15-19-58.jpg](%E5%9F%BA%E4%BA%8EPyTorch%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%201c33f1da2cf58096be53cae8b48e3b6e/Snipaste_2025-03-27_15-19-58.jpg)
    

---

### 1.3 数据交互

在导入并存储数据后，我们需要一种数据访问与交互的方式，以方便之后的模型训练，Pytorch提供了`DataLoader`类，用于在模型训练和评估期间导航数据集实例，封装给定的dataset和其他参数

它提供了非常方便的数据加载方式，通常用于将数据集（如 `Dataset` 类的实例）转换成一个可迭代的批次数据流。其主要作用是：

- **批量数据加载**
    
    `DataLoader` 可以根据指定的批次大小（`batch_size`）自动从数据集中按批次加载数据。这对于深度学习中的训练和验证非常重要，因为通常我们会一次处理大量的数据，而不是一次性将整个数据集加载到内存中。
    
- **数据随机化**
    
    `DataLoader` 可以通过设置 `shuffle=True`，在每个 epoch 开始时随机打乱数据集。数据的随机化可以帮助提升训练过程中的模型泛化能力，避免模型学习到数据中的某些顺序或规律。
    
- **并行数据加载**
    
    `DataLoader` 支持多线程并行加载数据。通过设置 `num_workers` 参数，可以指定用于加载数据的线程数。这会加速数据的加载过程，尤其是在处理大型数据集时，能够显著减少模型训练的等待时间。
    

> 训练神经网络中最基本的三个概念：**Epoch, Batch, Iteration**
> 
> 
> 每个 Epoch 需要完成的 Batch 个数等于 Iteration 个数
> 
> ![v2-18f7f8e6cf5c827217f076483f16e986_r.png](%E5%9F%BA%E4%BA%8EPyTorch%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%201c33f1da2cf58096be53cae8b48e3b6e/v2-18f7f8e6cf5c827217f076483f16e986_r.png)
> 

我们用实例说明`DataLoader`的作用：

```python
# 划分训练集与测试集
train, test = dataset.get_splits()

# 为训练集和测试集创建 DataLoader，此时的train_dl等是DataLoader实例
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=1024, shuffle=False)
print(len(train_dl.dataset), len(test_dl.dataset))

# 在本例中，train_dl 的 batch_size 为 32，数据将随机排序。让我们来查看一下 train_dl
n_inputs = len(train_dl)
for i, (inputs, targets) in enumerate(train_dl):  
    print(f'第 {i} 个 batch 有 {len(inputs)} 个数据，其中输入矩阵的形状是 {inputs.shape}，输出矩阵的形状是 {targets.shape}')
print(f'共有 {n_inputs} 个 batches')
```

- 输出：

![Snipaste_2025-03-27_16-03-48.jpg](%E5%9F%BA%E4%BA%8EPyTorch%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%201c33f1da2cf58096be53cae8b48e3b6e/Snipaste_2025-03-27_16-03-48.jpg)

可以看出，DataLoader维护了传入的dataset实例，通过给定的batch_size和shuffle属性，选择内部dataset的一部分数据进行每批次的输入输出

> **enumerate()函数**是 Python 内置函数，用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个有索引的序列，同时列出数据和数据下标。多用在 for 循环中。
例如：
> 
> 
> ```python
> seasons = [('Spring', 'Green'), 
>            ('Summer', 'Red'), 
>            ('Fall', 'Yellow'), 
>            ('Winter', 'White')
>            ]
> print(list(enumerate(seasons, start=1)))  # start 参数不填则默认从 0 开始
> print('--------')
> 
> # 再在 for 循环中看看 enumerate 函数的效果
> # season和color是我们自定义的变量名，与数组中相应位置的值一一对应，只是方便理解
> for i, (season, color) in enumerate(seasons, start=1):
>     print(f'My impression {i} about {season} is {color}.')
> ```
> 
> 输出：
> 
> ![Snipaste_2025-03-27_16-14-07.jpg](%E5%9F%BA%E4%BA%8EPyTorch%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%201c33f1da2cf58096be53cae8b48e3b6e/Snipaste_2025-03-27_16-14-07.jpg)
> 

---

## 2.定义模型

### 2.1 常见模型