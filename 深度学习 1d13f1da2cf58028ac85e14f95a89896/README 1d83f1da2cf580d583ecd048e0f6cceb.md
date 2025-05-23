# README

日期: 04/20/2025
状态: 进行中

<aside>
👓

@whiszk 

04/20/2025 

</aside>

经过约两周半的学习，从基础的图像质量评估指标到Swin Transformer这一前沿架构，以及相关的背景知识，我对计算机视觉领域有了初步了解与认识，以下是一些个人感悟：

1.图像质量评估指标的演化折射了什么

- 作为传统图像质量评估的基石，PSNR和SSIM为许多研究提供了量化标准，但本质还是单纯的统计学公式应用，充满**手工设计**的痕迹，显然没有人能肉眼识别像素值，这样的指标也无法真正刻画人类直觉。
- 因此LPIPS呼之欲出，利用神经网络来刻画深层特征，具有强数据驱动性，从表现来看，向成功刻画主观映像迈出了坚实一步，但这样的做法**可解释性较差**，很自然地提出疑问：堆参数能否解决一切问题呢？当下的大语言模型感觉自己很行，事实上也确实很好满足了9成的需求，但我们仍需谨慎看待不同任务的特殊性。

2.计算机视觉的本质性思考

- 计算机视觉的核心是让机器“看懂”图像，那么图像的本质又是什么？像素点是最终答案吗？好像一切一切的研究，都是基于像素值这个先决条件，进行各种黑箱操作，有时对模型性能的追求，对参数的不断调整，都开始脱离了研究问题本身，我就感觉在学习过程中难以把握原理，大家好像都在绽放灵感，鲜有实质性的逻辑推导（当然这个说起来容易，我上我不行）。
- 终于到Self attention的提出，为深度学习领域注入了些”仿生学“的意味，利用了注意力这个概念，贴合人类观察时的习惯，让每个token更有生命力；应用到计算机视觉领域，同样取得了很大成功，称**认知科学和神经科学**为计算机视觉的重要源泉应该不为过，是值得探索的交叉领域。
- 结合前段时间的小米智驾事故，不得不说生命还是难以托付给黑箱与概率学。
1. 关于矢量图的想法
- 我们都知道图片分为位图与矢量图，目前对于矢量图的研究深度却远不如位图，其中也有数据量与应用场景因素的影响。我们总是要对多通道位图进行向量化、序列化的预处理，与之相比，矢量图天生就具有结构化表示的优势，会不会更容易在特征提取过程中保留语义呢？