# Swin Transformer Block

<aside>
👓

@whiszk 
04/18/2025 

</aside>

---

## 一、Block主体结构

block都是**成对出现**，一个stage中只会有偶数个block，因为W-MSA和SW—MSA需要交替应用

![一组Swin Transformer Block
       （visio绘制）](Swin%20Transformer%20Block%201d93f1da2cf5802191aac4a5f286caad/%E7%BB%98%E5%9B%BE2.jpg)

一组Swin Transformer Block
       （visio绘制）

代码包含以下子模块：

- W-MSA**或**SW-MSA，通过`WindowAttention`调用
- `Window_partition`，简单函数，`(B, H, W, C)`->`(num_windows*B, window_size, window_size, C)`
- LayerNorm
- mlp
- 残差连接

```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # 输入图片尺寸小于设置的msa窗口大小时，直接使用图片尺寸作为msa窗口大小，且不移位
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
					
				 # 移位步长大于0，表示这个block要使用sw-msa，此处计算生成一个注意力掩码
**A**       if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            
            # 为每个区域分配唯一 ID
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

						 # 根据id，生成注意力mask矩阵，与forward中的循环移位搭配使用
 **B**          mask_windows = window_partition(img_mask, self.window_size)  # 窗口数量, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # 展平
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process # 计算优化开关，需要cuda支持

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x # 保存原始输入，用于后续残差连接
        x = self.norm1(x) # 先对通道维度C进行layernorm
        x = x.view(B, H, W, C) # 将patch序列恢复为空间排列的二维网格，便于后续的窗口划分和移位操作

        # 循环移位，向左向上移动
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
				
				 # 展平，nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 传入W-MSA/SW-MSA模块
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

				 # 恢复展平前的窗口形状
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # 恢复位移前的patch位置
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x) # 第一次残差连接

        # 一行完成三个操作：第二次layernorm，mlp，残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
```

---

### 1.循环移位的逻辑链

窗口移动是以patch移动实现的(相对运动)，以`W，H=6` `windowsize=3` `shiftsize = 1`举例：

原代码中的**A部分**：

- 在`init`函数中，先通过slice与for来**分配区域id**，原来在同一个window中，或者在两个不同window但紧挨（遵循边界，不进行空间跨越）着的patch，会被分配到相同的id
- 得到的`img_mask` 是一个 `[1, 6, 6, 1]` 的张量，内容为：
    
    ```markdown
    [
      [[[0]], [[0]], [[0]], [[1]], [[1]], [[2]]],  # 第0行
      [[[0]], [[0]], [[0]], [[1]], [[1]], [[2]]],  # 第1行
      [[[0]], [[0]], [[0]], [[1]], [[1]], [[2]]],  # 第2行
      [[[3]], [[3]], [[3]], [[4]], [[4]], [[5]]],  # 第3行
      [[[3]], [[3]], [[3]], [[4]], [[4]], [[5]]],  # 第4行
      [[[6]], [[6]], [[6]], [[7]], [[7]], [[8]]]   # 第5行
    ]
    ```
    
    可视化展示：左边的部分代表W-MSA的窗口划分与注意力计算，对于**右边的部分**，首先明确只有在同一个window中才可以互相attention，进一步的，在单个window中，只有**区域id相同的patch**（红框圈起来的部分，与上文区域代码对应），才可以互相attention。相当于在大window中引入了小window。
    
    ![                    循环移位与mask注意力计算示意图（ppt绘制）](Swin%20Transformer%20Block%201d93f1da2cf5802191aac4a5f286caad/%E6%BC%94%E7%A4%BA%E6%96%87%E7%A8%BF1.jpg)
    
                        循环移位与mask注意力计算示意图（ppt绘制）
    

对这样的划分方式，我们可以这样理解：

- 在一次wmsa过后，我们希望不同window中相邻的patch关注一下彼此，**让人为的窗口划分不那么生硬**，但是原来那些不挨着的，就没必要交流了，这就完成了一轮swmsa；
- 再进入下一轮wmsa时，因为window中的部分patch已经互相交流过了，他们不会过于陌生，对整个window做attention时就不会显得突兀，所以**实际上的滑动窗口，有两步走的过程**。
- 需要明确的是：无论进行多少次移位，所有patch的**绝对物理坐标**始终不变（降采样不算），坐标位置在预处理阶段就确定了，移位只是暂时的，前一轮wmsa中，那些不相邻且不在一个window中的patch，不会在后二轮wmsa中关注彼此，信息的传递永远是间接的，因为移位只是传递信息的手段，是多次移位让相隔很远的patch间接交流，而不是疯狂打乱patch位置来起到交流的结果。
- 打个比方，自习教室中位于两个角落的人，通过中间人准确传递信息，而**不是离开座位交流或者直接大喊。**

---

原代码中的**B部分**：

- 之前的区域划分还需要处理才能用于注意力计算，下文为每个patch提供了长度为9的注意力序列，包含了所属窗口的所有patch，如果区域id相同，才会正常计算注意力
    
    ```python
    # partition之后的维度： 窗口数量, window_size, window_size, 1
    mask_windows = window_partition(img_mask, self.window_size)  
    
    mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
    '''
    展平之后：
    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 2., 1., 1., 2., 1., 1., 2.],
            [3., 3., 3., 3., 3., 3., 6., 6., 6.],
            [4., 4., 5., 4., 4., 5., 7., 7., 8.]])
    '''
    
    # unsqueeze可在指定位置插入一个新维度，一般搭配广播机制使用
    #	所有位置对进行相减，得到window_size*2为高宽的矩阵
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    
    # 相减结果不为0的，置-100，这样在softmax时会被直接忽略，只有来自同区域的patch允许计算注意力
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    ```
    

---

### 2.位移与恢复位移

`x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))`

注意到在传入attention模块得到结果后，恢复了位移前的patch位置，更明确了[这一点](Swin%20Transformer%20Block%201d93f1da2cf5802191aac4a5f286caad.md)

---

## **二、W-MSA和SW-MSA**

- 同时支持**W-MSA**和**SW-MSA**的模块
    - W-MSA阶段：窗口内自由交流（局部信息整合）
    - SW-MSA阶段：让**物理相邻但被窗口分割**的patch建立连接（非常重要）
- 对输入图像做一个固定大小窗口（如 `7×7`）划分，每个窗口内做 Self-Attention，与Transformer的区别是：注意力不再全局计算，只在窗口内部进行，计算量大幅降低，且swint的self attention会实时感知位置，通过引入**相对位置偏置表**实现：
    - 如果两个位置对的相对坐标差（Δh, Δw）相同，那么它们的相对位置偏置也会相同，即对所有头的注意力计算，都采用同一套位置偏移量
    - 每个注意力头都有一套独立的相对位置偏置参数，参数总维度为 **`[位置对总数, 注意力头数]`**，用于补充空间结构信息，使注意力机制具备位置信息的建模能力
    

```python
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim # patch的通道数，即C
        self.window_size = window_size  # attention window的高与宽
        self.num_heads = num_heads # 注意力头的数量
        head_dim = dim // nudm_heads
        self.scale = qk_scale or head_dim ** -0.5 # 后者是默认的\frac{1}{\sqrt{d_k}}

        # 定义相对位置偏置参数，维度是[2*Windowh-1 * 2*Windoww-1, num_heads]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 构造相对位置位置索引表（不可学习）：用于索引偏置表
        coords_h = torch.arange(self.window_size[0]) # 生成0->size-1的一维张量，为创建表格提供坐标
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 广播机制， 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index) # 保存为静态参数，不参与训练

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # 这个B_是所有图片中的总窗口数，是窗口划分之后的结果，B_=num_windows（一张图片的窗口数） * B
        B_, N, C = x.shape
        
        # 从线性层初始化多头qkv矩阵
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = q * self.scale # 相当于\sqrt{d_k}
        attn = (q @ k.transpose(-2, -1)) # QK^T，attn形状为[B_, num_heads, Wh*Ww, Wh*Ww]
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # 加上位置偏置
                
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn) # 对attn最后一个维度（即每个query对应的所有key）做Softmax
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x # 输出维度为[B_, N, C]

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N): # N为patch数
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
```

---

### 1.相对位置索引表的构造

- 构造**相对位置位置索引表**的过程中，`meshgrid`会返回两个矩阵，**`stack`**将这两个矩阵堆叠，得到**`coords`，**为了便于计算相对位置，还要再经过展平，之后的相对位置计算比较繁琐，其中应用了广播机制
- 这里直接给出`relative_position_index`的结果，比如输入坐标是两个`[0,1,2]`，则：
    
    ```python
     meshgrid结果：                      stack之后：        
    [[0, 0, 0],     [[0, 1, 2],        [
     [1, 1, 1],      [0, 1, 2],         [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # 高度坐标
     [2, 2, 2]]      [0, 1, 2]]         [[0, 1, 2], [0, 1, 2], [0, 1, 2]]   # 宽度坐标
    																			 [
    
    flatten展平之后：
    [
      [0,0,0,1,1,1,2,2,2],  # 所有位置的行坐标
      [0,1,2,0,1,2,0,1,2]    # 所有位置的列坐标，上下pair表示一个点
    ]			
    
    ……………
    
    relative_position_index：
    
    i\j	  (0,0)	(0,1)	(0,2)	(1,0)	(1,1)	(1,2)	(2,0)	(2,1)	(2,2)
    (0,0)	12	   11	   10	    7	    6	    5 	    2	    1	    0
    (0,1)	13	   12	   11	    8	    7	    6	    3	    2	    1
    (0,2)	14	   13	   12	    9	    8	    7	    4	    3	    2
    (1,0)	17	   16	   15	    12	    11	    10	    7	    6	    5
    (1,1)	18	   17	   16	    13	    12	    11	    8	    7	    6
    (1,2)	19	   18	   17	    14	    13	    12	    9	    8	    7
    (2,0)	22	   21	   20	    17	    16	    15	    12	    11	    10
    (2,1)	23	   22	   21	    18	    17	    16	    13	    12	    11
    (2,2)	24	   23	   22	    19	    18	    17	    14	    13	    12			 
    ```
    

---

### 2.相对位置偏移参数

关于`self.relative_position_bias_table`的**参数数量**如何确定：

- 如果窗口大小是`7×7`，也就是窗口中有49个token，那么对任意token对，它们的横向相对位移范围是`-6 ~ +6`，共 `13` 种，这个13就是2*windowsize-1的结果，纵向也是一样，所以在二维空间中，一共有`13×13`种相对位置
- 因为我们只关心**相对位置**，不管token在窗口内的绝对位置是多少，只要`(i - j)`是一样的，它们的attention就用同一个bias
- 第二个维度是注意力头数量，每一个头关注patch的不同维度区域

---

### 3.attn与qkv

重要操作：`qkv = self.qkv(x).reshape(……).permute(……)`

- `self.qkv(x)`：使用线性映射，shape为**[B_, N, 3 × C]**，等待拆分
- `.reshape(B_, N, 3, self.num_heads, C // self.num_heads)`：为每个 head 都生成了一个`head_dim`的 Q/K/V向量，体现了每一个头关注patch的不同维度区域
- `.permute(2, 0, 3, 1, 4)`调换维度，变为**[3, B_, num_heads, N, head_dim]**
``这样就有q, k, v= qkv[0], qkv[1], qkv[2]

`attn = (q @ k.transpose(-2, -1))`，@是矩阵乘法，此处attn相当于$QK^T$，维度在最后输出时也是`[B_, num_heads, Wh*Ww, Wh*Ww]`，表示每个token对其他token的注意力分布

---

### 4.mask

传入的mask形状为`[总窗口数, 单窗口patch数*,* 单窗口patch数]`，调整attn的同时，将`mask`广播到`[1, nW, 1, Wh*Ww, Wh*Ww]`，-100的数值足以让softmax后的注意力权重归0

```python
if mask is not None:
    nW = mask.shape[0]  # 获取窗口数量
    
    # 将注意力分数变形为 [B, nW, num_heads, N, N] 以便与 mask 对齐
    attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    
    # 恢复形状 [B*nW, num_heads, N, N]
    attn = attn.view(-1, self.num_heads, N, N)
    
    # Softmax 处理（-100 的位置权重趋近于 0）
    attn = self.softmax(attn)
```

---

### 5.可训练参数

总结WindowAttention的可训练参数组成：

| 模块 | 参数 | 大小 | 含义 |
| --- | --- | --- | --- |
| QKV 映射 | `self.qkv` | `[C, 3C]` | 把每个 token 映射为 Q/K/V |
| 输出映射 | `self.proj` | `[C, C]` | 把多头输出融合回 C 维 |
| 相对位置偏置 | `self.relative_position_bias_table` | `[(2Wh-1)(2Ww-1), nH]` | 给每个 token 对加偏置，提高空间建模能力 |