# Transformer

<aside>
👓

@whiszk 
04/10/2025 

</aside>

> 参考博客与视频：
[一文了解Transformer全貌（图解Transformer）](https://www.zhihu.com/tardis/zm/art/600773858)
[Transformers explained visually 油管3B1B](https://www.youtube.com/watch?v=wjZofJX0v4M&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6)
（视频中有关于单词token在多维空间中分布概念的生动讲解，比如，我们在某个维度方向上，取出德国和日本的距离向量，加到寿司上，得到的空间点很可能是德国香肠，这样就很直观地理解在这个空间中，token是如何按照特征分布的，我们能以无穷丰富的观察角度来描述这些token的关系）
> 

---

## 一、概念解释

**Transformer** 是一种完全基于注意力机制（self-attention）的神经网络结构，用来建模序列数据之间的依赖关系，突破了 RNN 和 CNN 的**时间/空间限制**，于2017年在《Attention Is All You Need》这篇文章中首次提出，并最初用于解决文本序列建模任务（如翻译）。
下文讨论的transformer结构都是指该论文提出的原始结构。

结构主体为多层**Encoder**和**Decoder**，Decoder会结合前一个时间步的输出来给出当前时间步的输出，所以transformer在**推理**（区别于训练）时的输出是序列的，单个单个的。

![         transformer架构示意图（visio绘制）](Transformer%201d13f1da2cf580ac81c2ca39cf7c5ec3/8.jpg)

         transformer架构示意图（visio绘制）

---

## 二、输入处理

输入要先经过**Embedding**和**Positional Encoding**。

- Input Embedding 就是把离散的输入（比如词、字符、图像块等）转换成一个**连续的向量表示**，这个向量可以看作该token（对于文本来说，通常为单词、单词的一部分、或常见的字符组合等）在一个高维空间的坐标，词义相近的token坐标位置也相近，这样的数学表达便于神经网络进行后续处理。
    
    例如，对于**自然语言输入**：“I love cs”，通过一个Embedding矩阵被映射成三个固定维度的向量；对于**图像数据**：有一张维度为`[3, 224, 224]`的图片，可以通过16×16的patch来划分，得到196个`[3, 16, 16]`的patch，再展平每个patch，维度为3×16×16=768，最终得到196个768维的向量（vision transformer）。
    
- Positional Encoding，一般是对embedding的结果直接加上位置码，原论文采用的是三角函数，选择可以有很多

---

## 三、Encoder模块（提取特征）

原始论文提出的Transformer有多层相同的Encoder，每层包含以下模块（有先后顺序）：

- Multi-Head Self Attention
- Add & LayerNorm
- Feed Forward Network（FFN）
- Add & LayerNorm

关于什么encoder的结构是这样，原论文提出的结构并非所谓最优解，现在也有其他微调之后的架构，至于性能提升也许是仁者见仁

![image.png](Transformer%201d13f1da2cf580ac81c2ca39cf7c5ec3/image.png)

---

### 1.**Multi-Head Self Attention**

见[Self-attention](Self-attention%201d13f1da2cf580ddb86ff5ae7e005e6d.md) 中的Multi-Head Self Attention，这个过程可以认为是所有token在互相交谈，以确定自己在这个输入序列中的具体角色，比如为自己增加一些场景词义。

---

### **2.Add & LayerNorm**

add的过程是**Residual Connection**，将输入与Self Attention的输出**直接相加**。引入这个残差连接，是为了解决深层神经网络的**退化问题**。

- 有关深层神经网络的退化问题：
我们往往认为越深的网络理论上表达能力越强，但事实上网络越深，参数越多，梯度越难正确更新，由于非线性组合过多，深层网络容易学出过度复杂、难以优化的函数，导致性能反而不如浅层神经网络。
- 残差连接如何解决这一问题呢？
    
    假设某一层有输入x，子层输出为$f(x)$，残差结构为：
    
    $$
    output = x + f(x)
    $$
    
    如果$f(x)≈0$，这层就“**基本没做事**”，输出就是输入：$x+0=x$，信息直接跳过去，相当于该神经元进行休眠，只起到传递信息的作用。
    
    换句话说，残差连接把网络的目标从直接学习一个复杂函数$H(x)$，变为学习一个残差函数 $F(x)$，再对输入进行一个简单的线性修正，增强了**特征在深层网络中的穿透能力**，同时保留了良好的非线性表达力与模型的训练稳定性。
    

---

该架构使用的归一化方式为**layer normalization**，两种常见归一化方式的区别如下：

| 名称 | 归一化对象 | 均值方差的计算维度 |
| --- | --- | --- |
| **BatchNorm** | 一个 batch 的多个样本，在**每个通道**上统一归一化 | 沿着 batch 维度（比如 shape: `[B, C, H, W]`，归一化的是 BHW），对于每个通道，将这个通道中**所有样本的所有数据点**，标准化成均值为 0，方差为1 |
| **LayerNorm** | 每个样本自己，在**所有通道/特征**上做归一化 | 沿着特征维度（比如 shape: `[C]` 或 `[C, T]`），layer归一化的结果就是，每个样本自身所有通道的数据点均值为0，方差为1 |

打个比方：

| 类比 | BatchNorm | LayerNorm |
| --- | --- | --- |
| 考试成绩标准化 | 全班每个学生某一科成绩一起标准化 | 每个学生所有科目的成绩自己标准化 |

关于layer归一化的**优势**：

- 不依赖batch size，适合小batch的场景
- 在NLP或语音任务中，输入常是变长的序列（不同样本长度不一样）。而LayerNorm只对一个 token的所有特征做归一化，不受序列长度影响
- BatchNorm在训练和测试阶段的行为不同（训练时用batch统计，测试时用累计均值方差）。而LayerNorm则没有这个差异，在训练和推理时行为完全一致，**部署更简单、推理更稳定**。

---

### **3.Feed Forward Network（FFN）**

FFN是Transformer中用于**逐位置、独立地**处理每个token特征的两层MLP，用于增强特征的非线性表达能力。这个过程中各token不再交流，可以认为是**对每个token单独提出问题**，根据回答来更好地刻画他们。
因为Multi-Head Self Attention的处理是线性的，难以捕捉输入间复杂的非线性关系，所以需要FFN帮助模型学到更复杂的映射关系。FFN 层由两个线性变换层和一个 ReLU 激活函数组成：

$$
FFN(x)=max(0,xW_1 +b_1)W_2+b_2
$$

其中：

- `xW1 + b1` 为第一个全连接层的计算公式
- `max(0, xW1 + b1)` 为 ReLU 的计算公式
- `max(0, xW1 + b1)W2 + b2` 则为第二个全连接层的计算公式
- 计算时每个输入互相独立，不再像上一层一样互相关注

---

## 四、Decoder模块（生成）

Decoder结构与Encoder十分相似，不同的是：

- 第一个Multi-Head Attention采用了**Masked**操作
- 第二个Multi-Head Attention的$K，V$ 矩阵使用Encoder的**编码信息矩阵**进行计算，而 $Q$ 使用上一个Decoder的输出计算

### 1.masked 和 shifted right

masked很好理解，之前的self attention中，各输入向量可以看到所有其他向量的k值，这在当前的翻译任务中是不允许的，我们只能根据已有的内容来得到输出，因为真正的推理场景下，不可能提前得知正确输入，所以就需要采取masked，如下图，各向量只能关注到自己之前的输入向量

![Snipaste_2025-04-11_19-54-30.jpg](Transformer%201d13f1da2cf580ac81c2ca39cf7c5ec3/Snipaste_2025-04-11_19-54-30.jpg)

但需要明确的是，“Shifted Right”操作**只存在于训练阶段**。

- 在**推理阶段**，我们一开始只给 Decoder 一个 `<sos>` 起始标志，Decoder 根据 Encoder 的输出生成第一个词，比如 `我`，然后我们把 `<sos> 我` 再喂回 Decoder，生成下一个词，如此循环，直到生成 `<eos>` 或达到最大长度，在生成的过程中，我们用 mask 确保各输入向量不能看到自己位置之后的内容。
- 但在**训练阶段**，一个词一个词生成的效率很低，这样调整模型太慢了，所以训练是 **“并行训练”**：一次性把整个目标序列拿出来学。这时就需要shifted right，比如目标句子是 `I love apples`，我们需要考察每个词的输出情况，所以构造出 **Decoder 输入：`<sos> 我 喜欢`**（右移一位），对应的标签（预测目标）是：`我 喜欢 苹果`，这就是 **Shifted Right**：构造出“前缀版本的目标序列”，一次性并行训练所有位置的预测。同时加上 Mask，防止模型在训练时看到“自己要预测的词“。

总结一下，Shifted Right 是为**训练**构造“生成前缀”的输入序列，Masked Self-Attention 则防止模型在**训练或推理**时“偷看”未来信息。两者配合，让 Transformer Decoder 实现了并行训练 + 自回归生成。

---

### 2.cross attention

在decoder的multi-head attention这个模块中，主要思想是：

- 让 Decoder 利用 Encoder 提取好的输入语义信息，来辅助当前的目标序列生成过程。
- **查询矩阵$Q$，**是上一层的mask self attention输出经过线性映射后得到的，表示的是 Decoder 当前已生成部分（即输出前缀）的语义表达需求
- **键矩阵$K$和值矩阵$V$，**来自**Encoder 最后输出**的隐层状态，表示原始输入序列的语义特征提取

首先$K，V$矩阵肯定来自于 Encoder，因为 Decoder 中没有完整的语义信息，然后查询矩阵来自于 Decoder 也很好理解，该查询矩阵的内涵是基于当前已生成的输出，**试图去捕捉“下一个最合适词汇”的语义表达需求，**”我当前已经有了这些信息，我看看你提取的语义中，什么最适合辅助生成我紧接着的输出“，它就像人类在翻译时，读到一半会问自己：“嗯，接下来我应该翻哪一个词最合适？”——这种“提问”的语义意图就是 Query。

![Snipaste_2025-04-11_18-46-51.jpg](Transformer%201d13f1da2cf580ac81c2ca39cf7c5ec3/Snipaste_2025-04-11_18-46-51.jpg)

---

## 五、模拟training过程

### 1.准备数据

假设一个训练样本是：

- 输入（源语言序列）：`X = [x₁, x₂, ..., xₙ]`（如：“你好吗？”）
- 目标输出（目标语言序列）：`Y = [y₁, y₂, ..., yₘ]`（如：“How are you?”）

我们构造：

- Encoder 输入：`X`
- Decoder 输入（Shifted Right）：`<sos> y₁, y₂, ..., yₘ₋₁`
- Decoder 输出目标标签：`y₁, y₂, ..., yₘ`

---

### 2.正向传播

1. **Encoder**：
    - 将输入序列 X 进行嵌入（Embedding）+ 位置编码（Positional Encoding）
    - 输入进入多个 Encoder Layer，通过 Multi-Head Self-Attention + FFN 编码出整句的上下文语义。
    - 最终输出一个与输入等长的语义向量序列，作为 Cross-Attention 的上下文基础。
2. **Decoder**：
    - 对目标句子 `Y` 做 Shifted Right 得到 Decoder 输入（训练阶段一次性输入所有 tokens）
    - 先做 **Masked** Multi-Head Self-Attention，防止看到当前或未来词；
    - 接着做 **Cross-Attention**：用上一步的结果作为 $Q$， Encoder 的最终层输出作为 $K$ 和 $V$，抽取源句的相关语义；
    - 再过 Feed Forward Network、Layer Norm、残差等模块。
3. **线性映射 + Softmax**：
    - Decoder 最后输出一个形状为 `[batch_size, seq_len, vocab_size]` 的张量；
    - 每个位置都是一个词的概率分布，选择最可能的词作为当前预测。
    

---

### 3.计算损失

- 对于 Decoder 输出的每一个 token，都有一个对应的“正确答案”（目标 token）。
- 使用交叉熵损失（Cross Entropy Loss）来衡量预测词分布与真实标签之间的差异。
- 最终对整个句子/整个 batch 的平均损失反向传播。