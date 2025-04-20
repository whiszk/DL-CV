# Swin Transformer

<aside>
👓

@whiszk

04/11/2025

</aside>

> 原始论文：
[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030)
参考视频：
[Swin Transformer 油管coffee break](https://www.youtube.com/watch?v=SndHALawoag&list=PLpZBeKTZRGPMddKHcsJAOIghV8MwzwQV6&index=5)
参考博客：
[详解Swin Transformer](https://blog.csdn.net/qq_39478403/article/details/120042232)
[图解Swin Transformer（结合代码）](https://zhuanlan.zhihu.com/p/367111046)
> 

---

## 一、概念解释

### 1.SwinT与标准T

**Swin Transformer**（Shifted Window Transformer）是一种**视觉领域**专用的Transformer模型变体，它保留了原始Transformer的核心机制（Self-Attention + FFN + 残差 + LayerNorm），但对其结构进行了适配改进，使其成为适合图像任务的backbone，可以替代ResNet、ViT等用于分类、检测、分割等任务。

它的核心思想是：在视觉任务中，引入一种**滑动窗口**(Shifted Windows) + **层次结构**(Hierarchical Structure)的Transformer架构，使其能够：

- 高效捕捉**局部上下文**信息（小窗口注意力）
- 又能逐步建构**全局语义**（多层级结构 + 窗口滑动）

| 特点 | 原始 Transformer（NLP） | Swin Transformer（CV） |
| --- | --- | --- |
| 任务类型 | 文本序列建模（如翻译） | 图像建模（如分类/检测） |
| 输入形式 | Token（词或字符）序列 | Patch（图像块）序列 |
| Attention 范围 | 全局（Global Attention） | **局部窗口**（Window-based Attention） |
| 特征结构 | 不变 | 层级结构（Hierarchical） |
| 位置编码 | 显式加上 | 位置隐含在窗口划分中 |
| 下采样 | 无 | 有（Patch Merging 实现金字塔结构） |

---

### 2.SwinT相比标准T的改动

窗口注意力（Window-based Attention）

- 在原始Transformer中，每个Token都可以与其他所有Token做Attention，代价是 $O(n^2)$，在图像中极其昂贵（比如一张`224×224`的图像有上千个patch）。
- Swin 将Attention限制在**小窗口中局部进行**（比如`7×7`），显著降低了计算复杂度。

Shifted Window（**滑动窗口**）机制

- 由于只在局部做Attention，会丢失全局联系。
- Swin引入“**滑动窗口机制**”来让不同窗口之间的信息流动，多次移位让相隔很远的patch间接交流，从而打破生硬的窗口划分。

> 点击[这里](Swin%20Transformer%201d23f1da2cf580378293d425f1832e01/Swin%20Transformer%20Block%201d93f1da2cf5802191aac4a5f286caad.md)，查看关于滑动窗口的难点提醒
> 

金字塔结构（**层级化**）

- 原始Transformer没有下采样，不像CNN那样有金字塔结构。
- Swin在每个阶段通过“Patch Merging”对图像块进行下采样，使得模型能逐步提取更大感受野的特征（类似ResNet、FPN的做法）。

---

## 二、结构一览

整个模型采取层次化的设计，主体结构为4个Stage，每个stage都会缩小输入特征图的分辨率，像CNN一样**逐层扩大感受野**

- 首先对输入的图像进行patch分解，随后进入各个stage
- stage1：patch经过展平与线性映射得到token，进入第一轮Swin Transformer Block
- stage2：patch merge的`2×2`邻接 + 线性层**压缩**，使patch的数量减少1/4，维度（扩大4倍又压缩一半）变为2C，再进入block
- 后面的stage同理

![                       swin transformer架构示意图（visio绘制）](Swin%20Transformer%201d23f1da2cf580378293d425f1832e01/%E7%BB%98%E5%9B%BE1.jpg)

                       swin transformer架构示意图（visio绘制）

各阶段**输入输出维度：**

| Stage | 输入大小 | 操作 | 输出大小 |
| --- | --- | --- | --- |
| 预处理阶段 | `[B, H, W, 3]` | Patch Partition + Linear Embedding | `[B, (H/4)*(W/4), C]` |
| Stage 1减去Linear Embedding | `[B, (H/4)*(W/4), C]` | Swin Block×2 | `[B, (H/4)*(W/4), C]` |
| Stage 2 | `[B, (H/4)*(W/4), C]` | Patch Merging + Swin Block×2 | `[B, (H/8)*(W/8), 2C]` |
| Stage 3 | `[B, (H/8)*(W/8), 2C]` | Patch Merging + Swin Block×6 | `[B, (H/16)*(W/16), 4C]` |
| Stage 4 | `[B, (H/16)*(W/16), 4C]` | Patch Merging + Swin Block×2 | `[B, (H/32)*(W/32), 8C]` |

注：表中的Patch Partition划分窗口大小为`4×4`，进出每个block时，patch的形状都会被**reshape**一次，便于窗口注意力的计算

---

## 三、输入预处理

与ViT类似，Swin Transformer的输入图像也需要被划分成Patch，但其做法更贴合CNN风格：

- **patch partition**：将图像划分为不重叠的Patch，例如，对于`256×256×3`的图像，可以划分为`56×56=3136`个`4×4×3`维度的patch。
- stage1中的**Linear Embedding**：把每个Patch展平，并线性映射为一个C维向量，C有多种选择，根据模型大小自定义，比如，swin-tiny的C为96，swin-large的C为192。

例如输入图像为`[H, W, 3]`，划分后得到`N= (H/4)×(W/4)`个C维patch，随后这些patch tokens经过Linear Embedding处理，再被馈入若干具有改进自注意力的Swin Transformer blocks

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # 单纯维度复制，变为(img_size, img_size)
        patch_size = to_2tuple(patch_size) # 同上
        
        # 垂直（高）和水平（宽）方向上的patch数
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] # patch总数

        self.in_chans = in_chans # 输入的通道数
        self.embed_dim = embed_dim # 希望得到的patch维度

				 # 卷积操作，卷积核大小与步长等于patch大小，转换举例：(N, 3, 224, 224)->(N, 96, 56, 56)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 创建归一化模块
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 尺寸检查
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # 对应patch partition + stage1中的Linear Embedding，
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        if self.norm is not None:
            x = self.norm(x)
        return x
        
		# flops() 函数的作用是：估算 PatchEmbed 模块中的计算量，可以理解为复杂度
    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
```

要点：

- 注意patch的划分的过程不仅仅只是数据维度从[高，宽，通道数]变为[patch数，patch维度]的简单转换，而是对原始像素值也进行了**卷积处理**，其中的卷积核参数是可学习的
- `self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)`
这个卷积操作对应流程图的**patch partition**，并提前完成了线性映射到C维向量的过程
- `x = self.proj(x).flatten(2).transpose(1, 2)`：
    - `.flatten(2)`把后两个维度 `[H', W']` 拉平成一个维度 → patch 数量
    - `.transpose(1, 2)`交换通道维和patch数量维：将 `[B, C, N]` → `[B, N, C]`
    - 最终 shape 是 `[B, 3136, 96]`

---

## 四、S**win Transformer Block**

由于内容较多，放在下面的子页中

[S**win Transformer Block**](Swin%20Transformer%201d23f1da2cf580378293d425f1832e01/Swin%20Transformer%20Block%201d93f1da2cf5802191aac4a5f286caad.md)

---

## 五、Patch Merging

作用相当于CNN里的pooling，Swim的下采样方式是：

- 将相邻的`2×2`patch拼接得到维度为`4*C`的大token
- 再用一个 Linear 层压缩为`2*C`

结果就是**图像尺寸减半，通道数翻倍**：`[B, H, W, C]` → `[B, H/2, W/2, 2C]` 

```python
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution # 输入的分辨率
        self.dim = dim # 输入通道数
        
        # 用于压缩的线性层，4C->2C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        
        # 归一化层，输入维度是 4C
        self.norm = norm_layer(4 * dim) 

    def forward(self, x):
        H, W = self.input_resolution # 通过top模块传入的，forward进来的x并没有这个信息
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C) # reshape操作，方便进行相邻的2×2合并

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  #变为[B H/2 W/2 4*C],在通道维度拼接
        x = x.view(B, -1, 4 * C)  #再次reshape为[B H/2*W/2 4*C]

        x = self.norm(x) # 归一化,默认是LayerNorm
        x = self.reduction(x) # 压缩，4C->2C

        return x

		# 可通过print(patch_merging)实体展示信息
    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim # 与view有关
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim # 压缩过程
        return flops
```

参数与要点：

- 需要注意的是，在进入**`PatchMerging`** 之前，patch的shape是**`[B, N, C]`**，但是**`PatchMerging` 内部自己又把它 reshape 回 `[B, H, W, C]`**，这样它才能进行空间上的2×2邻接patch。
    - **`B, L, C = x.shape`**从这里可以看出输入的形状
    - **`x = x.view(B, H, W, C)`**这个就是reshape操作，**`view`**是pytorch库函数
- 切片操作，**`[start:stop:step]`**，所以在每个**`2×2`**方块中，x0抽取的是左上角patch，x1抽取的是左下角，以此类推，最后用cat把这四个patch按通道维度拼接在一起，就从原来的4个C维patch，变为1个4C维patch
    - 关于拼接与cat函数简单示例：原图片中相邻四个pacth拼接
        
        ```
        拼接前：
        x0 = [[1, 2],    x1 = [[5, 6],    x2 = [[9, 10],   x3 = [[13, 14],
              [3, 4]]          [7, 8]]          [11, 12]]        [15, 16]]
              
        拼接后：
        x = [
            [[1, 5, 9, 13], [2, 6, 10, 14]],
            [[3, 7, 11, 15], [4, 8, 12, 16]]
        ]
        ```