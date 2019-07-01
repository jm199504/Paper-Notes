**Inception Version 1 - 4 论文研读**

1. Inception[**V1**]: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

2. Inception[**V2**]: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

3. Inception[**V3**]: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

4. Inception[**V4**]: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

**Inception v1/GoogLeNet**

2014年，Szegedy在《Going deeper with convolutions》论文中提出深度卷积神经网络 Inception，并基于该网络在 ILSVRC14 (大规模视觉识别竞赛)中达到了当时最好的分类和检测性能，其中提交的架构模型名为GoogLeNet，含一种22层网络的Inception的化身。

该架构的主要特点是改进了网络资源的利用率，该架构允许增加网络的深度和宽度，同时保持计算预算不变，为了优化质量，架构决策基于赫布（Hebbian）原则和多尺度处理。

**GoogLeNet即Inception的化身：**

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Inception%20v1-4/images/1.png">

**动机/现存问题:**

1.由于信息位置的巨大差异，为卷积操作选择合适的卷积核大小就比较困难。信息分布更全局性的图像偏好较大的卷积核，信息分布比较局部的图像偏好较小的卷积核。

2.非常深的网络更容易过拟合。将梯度更新传输到整个网络是很困难的。

3.简单地堆叠较大的卷积层非常消耗计算资源。

**解决办法:**

1.同一层级设置多个不同尺寸的滤波器（即网络本质更宽，而非更深）如Inception模块(a)，使用不同大小的滤波器（1×1，3×3，5×5），以及最大池化，子层级联传入下一层Inception。

该naive结构存在问题：每一层Inception module的filters参数量为所有分支上的总数和，多层 Inception 最终将导致 model的参数数量庞大，对计算资源有更大的依赖。

2.深度神经网络耗费大量计算资源，因此为了降低计算成本，考虑在3×3和5×5卷积前添加1×1的卷积层（①作为降维模块[在不损失模型特征表达能力的前提下减少模型参数(filter)降低模型复杂度]②提高网络表达能力），在不增加网络深度且不影响性能下增大网络宽度，注意：1×1卷积是在最大池化层的后一层。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Inception%20v1-4/images/2.png">

**GoogLeNet架构:**

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Inception%20v1-4/images/3.png">

然深达20+层的神经网络难以避免梯度消失的问题，为了防止梯度消失，引入了辅助分类器，对其中2个Inception输出作softmax操作，在计算辅助损失时，总损失为辅助损失和真实损失的加权和，每个辅助损失权重为0.3，辅助损失仅用于训练，而推断过程不使用。

**补充：**

1.GoogLeNet有Deepconcat拼接层：tf.concat(3,[],[])

2.完整的 GoogLeNet 结构在传统的卷积层和池化层后面引入了 Inception 结构，对比 AlexNet 虽然网络层数增加，但是参数数量减少的原因是绝大部分的参数集中在全连接层，最终在ImageNet取得6.67%成绩。

 
**Inception v2**

2015年，Ioffe在《Batch Normalization: Accelerating Deep Network Training byReducing Internal Covariate Shift》提出优化滤波器和Batch Normalization（正则化方法）。

**优化办法：**

1.Batch Normalization可以让大型卷积网络的训练速度加快很多倍，同时收敛后的分类准确率也可以得到大幅提高。BN 在用于神经网络某层时，会对每一个 mini-batch 数据的内部进行标准化（normalization）处理，使输出规范化到 N(0,1) 的正态分布，减少了 Internal Covariate Shift（内部神经元分布的改变），可以减少或者取消Dropout和LRN ( Local Response Normalization) 局部响应归一化层，可简化网络模型。

2.使用两个3×3滤波器代替5×5的滤波器

**补充：**

1.BN实现tf.image.per_image_standardization()

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Inception%20v1-4/images/4.png">


**Inception v3**

作者探索扩展网络的方法，考虑使用适当的分解卷积和正则化，在ILSVRC 2012 分类任务挑战赛实现了单帧评估 21.2% top-1 和 5.6% top-5 误差率，且模型总参数不超过250万。

**动机/现存问题:**

1.减少维度会造成信息损失

2.卷积的计算效率偏低（即运算速度偏低）

**解决办法:**

1.将5×5的卷积分解为两个3×3的卷积运算以提升计算速度（使用5×5的卷积在计算成本上是3×3的2.78倍），所以使用叠加两个3×3卷积在实际性能上会有所提升。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Inception%20v1-4/images/5.png">

2.尝试将n×n的卷积核尺寸分解为1×n和n×1的卷积，例如3×3卷积等价于1×3卷积再3×1卷积，其发现计算成本降低了33%。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Inception%20v1-4/images/6.png" width="500">

即：5×5滤波器转为两个3×3，且再分解为3×1与1×3的重叠卷积。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Inception%20v1-4/images/7.png">


**Inception v4**

2016年，《Inception-v4, Inception-ResNet andthe Impact of Residual Connections on Learning》

**动机：**

1.传统的网络架构中引入残差连接(即微软的残差网络ResNet)曾在 2015ILSVRC 挑战赛中获得当前最佳结果


**解决办法/尝试：**

1.将 Inception 架构和****残差连接(skip connect)****结合，经过实验证明，结合残差连接可以显著加速 Inception 的训练，尝试多种新型残差，显著提高了在 ILSVRC2012 分类任务挑战赛上的单帧识别性能

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Inception%20v1-4/images/8.png">


**总结**

**Inception V1**——构建了1x1、3x3、5x5的 conv 和3x3的 pooling 的**分支网络**，同时使用 **MLPConv** 和**全局平均池化**，扩宽卷积层网络宽度，增加了网络对尺度的适应性；


**Inception V2**——提出了 **Batch Normalization**，代替 **Dropout** 和 **LRN**，其正则化的效果让大型卷积网络的训练速度加快很多倍，同时收敛后的分类准确率也可以得到大幅提高，同时学习 **VGG** 使用两个3×3的卷积核代替5×5的卷积核，在降低参数量同时提高网络学习能力；


**Inception V3**——引入了 **Factorization**，将一个较大的二维卷积拆成两个较小的一维卷积，比如将3´3卷积拆成1´3卷积和3´1卷积，一方面节约了大量参数，加速运算并减轻了过拟合，同时增加了一层非线性扩展模型表达能力，除了在 **Inception Module** 中使用分支，还在分支中使用了分支（**Network In Network In Network**）；

**Inception V4**——研究了 **Inception Module** 结合 **Residual Connection**，结合 **ResNet** 可以极大地加速训练，同时极大提升性能，在构建 Inception-ResNet 网络同时，还设计了一个更深更优化的 Inception v4 模型，能达到相媲美的性能

 
