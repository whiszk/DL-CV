# Swin Transformer的目标检测应用

<aside>
👓

@whiszk 
04/19/2025 

</aside>

---

## 一、概念解释

### 1.目标检测

**目标检测**（Object Detection）是一类计算机视觉任务，核心目标是：在图像或视频中找出所有感兴趣的物体，并标记每个物体的类别和位置（边界框），即**定位 + 分类**。详见[目标检测](../%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%201db3f1da2cf5809ba8d8cf2d2f82fbaa.md)笔记

主流目标检测模型框架：

| 模型 | 类型 | 特点 |
| --- | --- | --- |
| Faster R-CNN | 两阶段 | 高精度 |
| YOLO 系列 | 单阶段 | 实时性能强 |
| RetinaNet | 单阶段 | 使用 Focal Loss 解决类不平衡 |
| DETR | Transformer 架构 | 无需手工设计 anchor |
| Swin + Cascade R-CNN | Swin 作为 backbone | 精度强、泛化强 |

模型输入及输出流程：

1. 输入：原始图像
2. **Backbone**：提取特征（如 ResNet 或 Swin）
3. Neck（可选）：特征融合（如FPN）
4. Detection Head：输出边界框 + 类别

---

### 2.backbone

输入的原始图像是单纯的像素矩阵，它本身对模型是“低级信息”，Backbone的任务就是**提取其中的视觉特征**。在空间上逐层下采样、在语义上逐层增强，从而提取语义信息（物体的形状、位置、纹理等）

常见backbone：

| 类型 | 代表模型 | 所属架构 |
| --- | --- | --- |
| 卷积网络 | ResNet, VGG, EfficientNet | CNN |
| Transformer 网络 | ViT, Swin Transformer | Vision Transformer |
| 混合结构 | ConvNeXt, MobileFormer | CNN + Attention |

---

### 3.Swin Transformer如何作为backbone

通过引入**局部窗口计算**和**层级式特征融合**，Swin Transformer显著提升了计算效率并适应多尺度目标检测任务，其[各stage输出](Swin%20Transformer%201d23f1da2cf580378293d425f1832e01.md)的feature map语义特征优秀且层级化显著。

具体优点为：

- 层次化特征图：通过逐步合并图像块（Patch Merging），生成**多尺度特征**（类似CNN的FPN），适合检测不同大小的目标。
- 窗口注意力（W-MSA）：将自注意力计算限制在局部窗口内，**减少计算复杂度**（从*O*(*n*2)降至*O*(*n*)），同时通过移位窗口（SW-MSA）实现跨窗口信息交互。
- **兼容性**：可直接替换CNN骨干（如ResNet），适配主流检测框架（如Faster R-CNN、RetinaNet）

---

## 二、目标检测应用

### 1.不同版本的Swin T

| 参数 | 含义 | Swin-T | Swin-S | Swin-B（base） | Swin-L |
| --- | --- | --- | --- | --- | --- |
| **Depth** | 每个 Stage 的 Block 数量 | [2, 2, 6, 2] | [2, 2, 18, 2] | [2, 2, 18, 2] | [2, 2, 18, 2] |
| **Embedding dim（C）** | Patch -> Token 后的维度 | 96 | 96 | 128 | 192 |
| **Model size** | 参数量（百万） | 28M | 50M | 88M | 197M |

---

### 2.应用收集

Swin T在**对象检测**方面的官方实现：

https://github.com/SwinTransformer/Swin-Transformer-Object-Detection，这是一个基于**MMDetection框架**的Swin Transformer实现，MMDetection是一个基于PyTorch的开源目标检测框架与工具箱，专门用于目标检测、实例分割和全景分割等任务。

swin t的应用基本是作为backbone嵌入到已有目标检测框架。

| 检测模型 | Swin 版本 | 数据集 | 应用场景 | 主要贡献或改进 | 代码链接 |
| --- | --- | --- | --- | --- | --- |
| Faster R-CNN | Tiny | COCO | 通用目标检测 | 提升 mAP 约 5%，可无缝替换 ResNet | [GitHub - SwinDetect](https://github.com/microsoft/Swin-Transformer) |
| Cascade Mask R-CNN | Base | COCO | 多尺度检测 | 更强层级特征表达力，mAP 提升至 50.0+ | 同上 |
| RetinaNet | Small | COCO | 单阶段检测 | 提升小目标检测精度，推理速度尚可 | 同上 |
| Mask R-CNN | Tiny/Base | COCO | 实时实例分割 | 分割与检测精度显著提升，可迁移至 Cityscapes 等数据集 | 同上 |
| YOLOS（Swin 变体） | Custom | COCO | Transformer 检测 | 使用纯 Transformer 架构进行目标检测，结构端到端 | [GitHub - YOLOS](https://github.com/hustvl/YOLOS) |