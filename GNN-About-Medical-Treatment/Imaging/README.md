# 成像检查（GNN）

## 1-EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks and Broad Learning System

### 作者

Xue-han Wang ; Tong Zhang ; Xiang-min Xu ; Long Chen ; Xiao-fen Xing ; C. L. Philip Chen

### 问题/背景

BDGLS在脑电图（electroencephalogram ，EEG）情感识别

### 方法

BDGLS模型使用差分熵（differential entropy (DE)）特征作为输入评估SEED，与支持向量机（SVM），深信念网络（DBN），图卷积神经网络网络（DCNN）和DGCNN作对比。

### 基本流程

1.图卷积神经网络：

1.1 定义无向连接图

1.2 定义傅里叶域中的图的谱过滤

1.3 定义k-1阶切雪比夫多项式

1.4 定义动态图卷积神经网络（Dynamical graph convolutional neural networks ，DGCNN）

1.5 定义损失函数

2.广泛学习系统：

2.1 随机生成输入数据到特征节点的空间

2.2 基于增强节点扩散到随机广泛空间

### 原始数据

SSJTU emotion EEG dataset (SEED)

### 数据规模

15个受试者（7男+8女）和每一位都被实验三次，每次观察15个电影短片（每个短片带有标签（积极，消极，中立）），即每一个受试者有45次小实验。

### 核心价值点（创新）

提出了广泛动态图学习系统（broad dynamical graph learning system (BDGLS)）可以处理EEG信号，通过整合动态图卷积神经网络（DGCNN）和广泛学习系统（BLS）的优势，能够在非欧几里德域上提取特征并随机生成节点寻找连接权重。

### 结果

BDGLS平均识别准确率：93.66%；

DBN平均识别准确率：86.08%；

SVM平均识别准确率：83.99%；

GCNN平均识别准确率：87.40%；

DGCNN平均识别准确率：90.40%

### 网络框图

​                                                  

### 未来工作

1.考虑选择更有区分性的特征

2.对于EEG结构数据设计特殊的网络0



## 2-EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks

### 作者

Tengfei Song ; Wenming Zheng ; Peng Song ; Zhen Cui

### 问题/背景

人机交互中情感分析具有极其重要的作用

### 方法

使用邻接矩阵表示和基于图建模多通道EEG特征输入到模型分类预测，与传统GCNN不同之处在于可以动态学习不同脑电图EEG通道间内在联系，提取更具有辨别性的特征。

### 基本流程

1.图的表示：参考下图（一）；涉及基于高斯核函数的KNN算法

2.光谱图过滤（也称图卷积）：涉及拉普拉斯矩阵

3.用于脑电图情感识别的DGCNN模型：参考下图（二）涉及k阶切雪比夫多项式

4.DGCNN算法：介绍损失函数；优化方法；伪代码

### 原始数据

SJTU EEG情感数据集(SEED)和DREAMER数据集

### 数据规模

15位受试者（7男性+8女性）

### 核心价值点（创新）

提出了基于动态图卷积神经网络（DGCNN）的多通道的EEG情感分析方法。

### 结果

SEED数据集交叉验证的平均准确率分别为：86.23％，84.54％和85.02％

### 网络框图

   

### 未来工作

EEG数据量偏小会使得限制深度学习网络模型的性能，即考虑提升数据集量

## 3-3D fully convolutional networks for co-segmentation of tumors on PET-CT images

### 作者

Zisha Zhong ; Yusung Kim ; Leixin Zhou ; Kristin Plichta ; Bryan Allen ; John Buatti ; Xiaodong Wu

### 问题/背景

图像分割：正电子发射断层扫描（Positron emission tomography）和计算断层扫描（computed tomography，PET-CT）双模态成像可提供关键诊断现代癌症诊断和治疗的信息。在基于PET-CT的计算机辅助肿瘤阅读和解释中，自动准确肿瘤描绘是非常重要的。

### 方法

1.医学图像分割数据预处理：图像配准（image registration），空间重采样（spatial resampling），图像强度值阈值处理（image intensity value thresholding）等

2.分别对PET和CT单独训练学习到更加的判别特征来生成肿瘤/非肿瘤的概率图

3.基于图切割的共分割模型（the graph based co-segmentation model）中结合PET和CT上的两个概率图来产生最终的肿瘤分割结果

### 基本流程

1.医学图像分割数据预处理：图像配准（image registration），空间重采样（spatial resampling），图像强度值阈值处理（image intensity value thresholding）等

2.分别对PET和CT单独训练学习到更加的判别特征来生成肿瘤/非肿瘤的概率图

3.基于图切割的共分割模型（the graph based co-segmentation model）中结合PET和CT上的两个概率图来产生最终的肿瘤分割结果

### 原始数据

32 PET-CT图像数据集

### 数据规模

32位肺癌患者的PET-CT扫描图

### 核心价值点（创新）

提出了一种基于图分割的共分割模型以及结合全连接网络（fully convolutional networks，FCN）和语义识别框架semantic segmentation framework (3D-UNet)的肺肿瘤分割方法。

### 结果

   

### 网络框图

   

### 未来工作

无