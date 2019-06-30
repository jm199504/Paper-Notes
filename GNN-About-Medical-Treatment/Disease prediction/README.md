# 疾病预测论文梳理（GNN）

## 1-Distance Metric Learning using Graph Convolutional Networks: Application to Functional Brain Networks

### 作者

Sofia Ira Ktena, Sarah Parisot, Enzo Ferrante, Martin Rajchl, Matthew Lee,Ben Glocker, and Daniel Rueckert

### 问题/背景

在计算机视觉和模式识别中，评估图形间相似性是非常重要的研究问题，其中距离和相似度选择因应用而异且难以获取最优值。

### 基本流程

基于多项滤波器（polynomial filters）使用孪生/暹罗卷积神经网络（siamese graph convolutional neural network）应用于不规则图形，网络学习图形的潜在性表达，使用全局损失函数使正则化效果更强，其网络对异常值的鲁棒性更强，其中损失函数是最大化预测为同类的相似程度，最小化预测为不同类的相似程度，最小化预测为同类和不同类的方差。

1.构建归一化拉普拉斯矩阵（The normalised graph Laplacian）表示图形结构

2.使用图傅里叶变换为了定义图卷积（信号c与滤波器乘积）

3.使用k局部滤波器降低计算时间复杂度（近似理解为K阶切比雪夫多项式中的截断扩展）

4.GCN中每层输出特征映射为滤波器*拉普拉斯矩阵*输入特征映射之和

### 原始数据

Autism Brain Imaging Data Exchange (ABIDE) from fMRI data

### 核心价值点（创新）

利用图卷积神经网络和图光谱理论（spectral graph theory）度量已知节点对应关系的不规则图形间相似度。应用实例：将神经元间通路或大脑区域中功能连接被构建成图形，定义相似度函数可以预测脑部相关疾病。

### 结果

与传统的距离度量k-nn分类器（基于欧几里得距离）相比的性能提高了11.9%。

### 网络框图

​                                                  

### 未来工作

1.论文选用网络结构相对简单，可以考虑更复杂网络提升性能

2.考虑使用自编码和对抗性学习低维的连接网络表示图

 

## 2-Spectral Graph Convolutions for Population-based Disease Prediction

### 作者

Sarah Parisot⋆, Sofia Ira Ktena, Enzo Ferrante, Matthew Lee,Ricardo Guerrerro Moreno, Ben Glocker, and Daniel Rueckert

### 问题/背景

基于个体本身的特征及个体与潜在大群体的关联程度进行疾病预测。

### 基本流程

带有基于图像特征（fMRI、MRI）和非图像特征（年龄和采集位置）及带有表征信息的边（节点间相似性）组成的大群体的稀疏图，基于部分带有标签的节点通过图光谱卷积方法预测无标签的节点标签实现半监督学习的疾病预测。

1.数据选择：图像数据：静息状态fMRI（功能性磁共振成像，functional magnetic resonance imaging)）、结构MRI（磁共振成像，Magnetic Resonance Imaging）；非图像数据：年龄、性别、acquisition site（采集位置）

2.ABIDE数据预处理：选用针对ABIDE公开的数据处理办法，参考网站：http://preprocessed-connectomes-project.org/abide/

3.使用脊分类器（a ridge classifier）对节点的特征向量降维

4.ADNI数据预处理：简单选择138段脑结构（数据来源：http://adni.loni.usc.edu/）

5.图像数据相似度量：Sim(Sv, Sw)；非图像数据度量：ρ(Mh(v), Mh(w))

6.ABIDE数据的非图像特征选择：年龄和采集位置；图像特征选择rs-fMRI连接网络

7.对于不规则图形需定义局部图过滤器，基于图光谱理论和使用图信号处理（graph signal processing ，GSP）所提供工具构建CNNs

8.光谱图卷积（spectral graph convolutions）利用了傅里叶域的乘法，图傅里叶变换类比于欧几里域拉普拉斯特征函数

9.拉普拉斯归一化图（normalised graph Laplacian）表示图

10.推进Defferrard的工作，滤波器选择多项式滤波器（优势1：严格受限于空间；优势2：降低了卷积运算复杂度），可使用切雪比夫多项式中的截断扩张（truncated expansion）来近似这种滤波器

其他：使用ReLu提升非线性程度，输出层采用softmax函数，交叉熵为其损失函数。

### 原始数据

1.Alzheimer’s Disease Neuroimaging Initiative (ADNI) database

2.Autism Brain Imaging Data Exchange (ABIDE) database as healthy or suffering from Autism Spectrum Disorders (ASD)

### 核心价值点（创新）

挖掘图像和非图像的信息价值，考虑结合图像和非图像数据特征进行图光谱卷积。

### 结果

ABIDE准确率：69.5% （目前最高为66.8%） ；ADNI准确率：77%

### 网络框图

   

### 未来工作

1.构建更为有效的群体图（population graph）

2.考虑同一条边上带有多类信息，可以尝试用矢量代替标量

3.结合时间信息的纵向数据（个体在不同时间点上的观测值）

4.更丰富的特征向量（本文仅采取了2个非图像数据特征）

## 3-Convolutional neural networks for mesh-based parcellation of the cerebral cortex

### 作者

Guillem Cucurull1;2, Konrad Wagstyl3;4, Arantxa Casanova1;2, Petar Velickovic´1;5,Estrid Jakobsen4, Michal Drozdzal1;6, Adriana Romero1;6, Alan Evans4, Yoshua Bengio1

### 问题/背景

了解大脑皮层的组织结构和绘制大脑皮层分割图，重构结构磁共振扫描所产生的皮质表面，将大脑皮层分割作为网格分割任务，可利用网格研究脑组织健康和精神疾病的异常，更好地描述脑疾病。

### 方法

基于图卷积神经网络和构建图注意力机制网络实现大脑皮层的网格分割任务。

### 基本流程

1.介绍基准模型NodeMLP、NodeAVG、MeshMLP

2.构建图卷积网络Graph Convolutional Networks：

2.1 图光谱卷积（spectral convolutions）：傅里叶域的滤波器与信号的乘积

2.2 用拉普拉斯算子的切雪比夫k阶多项式截断扩展生成空间局部过滤器，可以不需要计算拉普拉斯图的特征分解

2.3 图卷积层输入图和输出图，输出结果取决于局部领域信息（距离中心最多k步）

2.4 通过对上层节点叠堆层来提升节点的邻域范围

3.构建图注意力机制网络Graph Attention Network (GAT)：

3.1 具有与图卷积网络相同的输入输出结构（不同点：隐式的卷积权重）

3.2 基于内容的自我关注机制实现这种网络结构（但仅限图中具有边缘的部分）

3.3 共享注意力机制计算卷积层权重

3.4 注意力系数表示结点j的特征对于结点i的重要性

4.NodeMLP、MeshMLP、GCN、GAT训练细节

4.1 GCN

4.1.1 输入为一个网格，输出为一个标签（对于每个网格中的所有结点）

4.1.2 拥有8层卷积和切雪比夫估计选用K=8

4.1.3 每层卷积含64结点和ReLu激活函数和每层后添加batch Normalization

4.1.4 使用average dice损失函数

4.2 GAT

4.2.1 输入为一个网格，输出为一个标签（对于每个网格中的所有结点）

4.2.2 K=8的注意力计算32个特征数量（共256个特征）

4.2.3 选用中心点在5-hop内的邻域计算注意力系数增大

### 原始数据

The Human Connectome Project dataset

### 数据规模

100个受试者，1个网格/per受试者，网格结点均已被注释，节点被划分为44区/45区/其他，所有网格来源于不同受试者，每一个网格有1195个结点（表示左半球大脑皮层的Broca区域），每一节点仅能被划分为一个区（6:1:1分别为训练;验证;测试集）

### 核心价值点（创新）

提出了图卷积神经网络和图注意力网络，挖掘潜在数据结构实现预测。

### 结果

NodeAVG的平均Jacc：49:9 ± 2:7

NodeMLP的平均Jacc：38:7 ± 2:8

MeshMLP的平均Jacc：51:8 ± 2:6

GCN的平均Jacc：58:1 ± 3:1

### 网络框图

未提供

### 未来工作

1.将度信息注入节点特征

2.增加额外功能特征改善模型区分效果

3.一个完整大脑皮层网格大概含有160万个节点，使用当前图卷积方法受限于GPU内存，可以考虑子采样降低网格分辨率或选择较小补丁进行处理

## 4-Multi-View Graph Convolutional Network and Its Applications on Neuroimage Analysis for Parkinson’s Disease

### 作者

Xi Zhang, Lifang He, Kun Chen, Yuan Luo, Jiayu Zhou, Fei Wang

### 问题/背景

帕金森氏病（pd）是普遍的神经退化性疾病，其中已有研究利用临床和生物标志物数据对局部放电预测，而神经影像学作为神经退化性疾病的另一种重要信息来源。

### 方法

基于成对学习策略和多视野图神经网络预测帕金森氏病。

### 基本流程

1.MVGCN整体流程

1.1 利用多视图图卷积网络获取神经影像的特征，输出为特征矩阵

1.2 聚合特征矩阵生成特征向量

1.3 softmax进行关系预测

1.4 训练过程：随机梯度优化和反向传播算法

1.5 GCN的输出为M维特征矩阵

2.MVGCN部分流程

Shuman等人表示谱域的图卷积的广泛性，GCN可以有效地对非线性关系样本进行建模，有更强的能力挖掘图特征，本文提出的MVGCN有以下2个特性:

2.1 多图跨视野的卷积操作

2.2 多视野图组合池化

### 原始数据

Parkinsons Progression Markers Initiative (PPMI)

### 数据规模

754个受试者（596PD+158HC）

### 核心价值点（创新）

提出成对学习策略（the pairwise learning strategies），即一种GCN网络和将多种形式脑图像融合在关系预测，进行预测PD病例。

### 结果

GCN的ACU：0:9537±0:0587

PCA的ACU：0:6443±0:0223

### 网络框图

   

图一：MVGCN整体框架

   

图二：MVGCN部分流程

### 未来工作

纯粹数据驱动，未利用临床领域知识、电子健康记录等临床数据，考虑融入临床数据

## 5-Bootstrapping Graph Convolutional Neural Networks for Autism Spectrum Disorder Classification

### 作者

Rushil Anirudh and Jayaraman J. Thiagarajan

### 问题/背景

分析自闭症谱系障碍

### 方法

基于集成方法的图卷积神经网络分析自闭症谱系障碍

### 基本流程

1.图结构：

特征图.性别、位置；噪声图eg.30%的边dropped;原生图(Navtice graph)邻接矩阵，特征图中若两个个体间性别相同则性别分数:s(sex) =λ1 > 1;采集位置s(site) =λ2 > 1

2.GCN：

其中光谱方法可分为图的显式谱表示和使用空间邻域非谱表示，基于局部切雪比夫多项式，使用一阶局部近似图卷积

3.集成学习：

图的集成（约20图）和edge-drop率（20%-30%）

其中：网络结构：

神经网络有3层，每层含16个神经单元；学习率0.005；dropout为0.3；3阶切雪比夫多项式近似图傅里叶变换；递归消除特征减少特征空间至2000个；epoch为200

### 原始数据

ABIDE数据集

### 数据规模

1112个受试者（其中含有20个采集位置）

### 核心价值点（创新）

提出了利用弱训练好的G-CNNs的集成图卷积神经网络（a bootstrapped version of graph convolutional neural networks，G-CNNs），可以降低模型在结构选择的灵敏性。

### 结果

本文所提出模型的准确率70.86%

### 网络框图

   

### 未来工作

1.考虑随机集合改进模型预测效果的理论依据

2.通过随机二元图扩展该思想

3.考虑集成阶段融入隐藏层

## 6-Multi Layered-Parallel Graph Convolutional Network (ML-PGCN) for Disease Prediction

### 作者

Anees Kazi, Shadi Albarqouni, Karsten Kortuem, Nassir Navab

### 问题/背景

疾病预测分类问题

### 方法

多层平行图卷积网络（层数=2）用于二分类问题，使用结构信息分别构建亲和矩阵（affinity matrix）和结合邻域图（neighborhood graph），不同于传统的集成方法，亲和图中每一个间接结构数据（年龄、性别、体重、body-mask）元素带有样本空间的邻接关系和统计特性，N个受试者（含有d维特征向量和m个元素），采取并行GCN实现分类问题。

### 基本流程

1.构建亲和矩阵结构（Affinity Graph Construction）

2.构建排序层（Ranking Layer）

3.介绍目标函数（Objective function）

### 原始数据

1.ABIDE

2.Shenzhen CXR Database；

### 数据规模

1.ABIDE：871个受试者（468个健康人+403个ASD患者）

2.Shenzhen CXR Database：662个受试者（326个健康人+336个患者）

### 核心价值点（创新）

本文所提模型轻巧和快速，将电子健康记录的结构数据作为成像数据的补充，在图卷积网络中添加了加权层，探索结构数据中的关系来衡量结构数据的各元素对潜在疾病的影响，提出了可以合并每个图形信息的模型，且该模型可并行，引入了一个根据它的预测任务可自学习加权元信息。

### 结果

相对基线分别提升了5.31 %和8.15 %；ROC曲线面积分别增加4.96 %和10.36%

### 网络框图

   

### 未来工作

无