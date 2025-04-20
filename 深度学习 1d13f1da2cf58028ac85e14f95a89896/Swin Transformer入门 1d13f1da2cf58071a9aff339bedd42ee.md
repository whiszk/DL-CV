# Swin Transformer入门

日期: 04/12/2025
状态: 完成

> 参考视频：
> 
> 
> [台大李宏毅讲解 self-attention Transformer](https://www.bilibili.com/video/BV1Xp4y1b7ih?spm_id_from=333.788.videopod.episodes&vd_source=98a6c96b95d562bfcdc24c1c16644dff)
> 
> [self-attention Transformer 公式推导与矩阵变化](https://www.bilibili.com/video/BV1q3411U7Hi/?share_source=copy_web&vd_source=31ec93f79296bab4585039b85cf22ea6)
> 

[Self-attention](Swin%20Transformer%E5%85%A5%E9%97%A8%201d13f1da2cf58071a9aff339bedd42ee/Self-attention%201d13f1da2cf580ddb86ff5ae7e005e6d.md)

[Transformer](Swin%20Transformer%E5%85%A5%E9%97%A8%201d13f1da2cf58071a9aff339bedd42ee/Transformer%201d13f1da2cf580ac81c2ca39cf7c5ec3.md)

> Vision Transformer是将原始Transformer直接用于图像的首次尝试，它把单词换成了图像块patch作为token，复用了Transformer的Encoder结构来提取图像特征，从而完成分类任务，实际上的结构创新不大。
而Swin Transformer则是在ViT的基础上，通过引入**窗口注意力**和**层级结构**使其更适合图像任务，基本只沿用了transformer的msa，其他模块的创新程度高，所以笔记省去了vit的内容，只记录swint。
> 

[Swin Transformer](Swin%20Transformer%E5%85%A5%E9%97%A8%201d13f1da2cf58071a9aff339bedd42ee/Swin%20Transformer%201d23f1da2cf580378293d425f1832e01.md)

> 关于其**目标检测应用**
> 

[Swin Transformer的目标检测应用](Swin%20Transformer%E5%85%A5%E9%97%A8%201d13f1da2cf58071a9aff339bedd42ee/Swin%20Transformer%E7%9A%84%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E5%BA%94%E7%94%A8%201db3f1da2cf580b6b9fbe59593be3876.md)