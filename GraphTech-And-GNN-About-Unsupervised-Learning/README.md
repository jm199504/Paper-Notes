## 图技术及图神经网络相关的无监督学习算法

1.社区发现算法

传统算法：CPM；LPA；Fast Unfolding；HANP；LFM；SLPA；BMLPA；COPRA

思想：以网络结构信息作为聚类标准，以模块度作为优化目标

局限性：仅利用了网络结构信息，没有涉及节点自身特征

2.DeepWalk

思想：基于NLP对节点随机游走构建节点的网络结构信息向量表示

局限性：仅利用了网络结构信息，没有涉及节点自身特征

优势：支持在线学习可扩展，并行计算

3.谱聚类

思想：基于图论的聚类方法

局限性：传统的谱聚类仅利用了网络结构信息，没有涉及节点自身特征

改进：使用节点间的特征相似度代替原有连通设为1的邻接矩阵可以结合节点特征和网络结构信息

4.LINE

思想：利用一阶相似度和二阶相似度对网络节点表达

局限性：仅利用了网络结构信息，没有涉及节点自身特征

优势：挖掘更多的网络节点相似性度量标准

5.LargeVis

思想：基于LINE的高维数据可视化算法（对LINE算法的改进）

6.图自编码GAE和变分图自编码VGAE

思想：基于图神经网络对网络节点表达

优势：学习了网络节点的自身特征和网络结构信息

代码：https://github.com/tkipf/gae

7.反正则化图自编码器ARGA和反正则化变分自编码器ARVGA

思想：基于图神经网络对网络结构重构

优势：在重构过程中不断优化网络节点的表达

代码：https://github.com/Ruiqi-Hu/ARGA

	•社区发现数据集下载：

	•（1）Zachary karate club

	•Zachary 网络是通过对一个美国大学空手道俱乐部进行观测而构建出的一个社会网络.网络包含 34 个节点和 78 条边,其中个体表示俱乐部中的成员,而边表示成员之间存在的友谊关系.空手道俱乐部网络已经成为复杂网络社区结构探测中的一个经典问题[1]。

	•链接：<http://www-personal.umich.edu/~mejn/netdata/karate.zip>

	•（2）


**调研细节**

**1.图技术中的无监督学习算法：**

（1）社区发现（Community Detection）：

特点：以网络结构信息作为聚类标准，以模块度作为优化目标，如经典算法：LPA、CPM、HANP等

内容：9种经典社区发现算法

- CPM派系过滤算法

- - k派系表示所有节点两两相连(派系=完全子图)

  - k派系相邻：2个不同k派系共享k-1个节点

  - k派系连通：1个k派系可以通过若干个相邻k派系到达另1个k派系

  - 步骤：

  - - 1.寻找大小为k的完全子图
    - 2.每个完全子图定义为一个节点，建立重叠矩阵[对称]（对角线元素均为k，派系间值为共享节点数）
    - 3.将重叠矩阵中非对角线元素小于k-1值置为0
    - 4.划分社区

- LPA

- - 步骤：

  - - 1.所有节点初始化唯一标签

    - 2.基于邻节点最多标签更新节点（最多个数标签不唯一时随机）

    - 备注：涉及同步更新和异步更新：

    - - 同步更新：节点z在第t次迭代依据于邻居节点第t-1次label
      - 异步更新：节点z在第t次迭代依据于邻居节点第t次label和邻居节点第t-1次label（部分更新）

- Fast Unfolding

- - 基于模块化modularity       optimization启发式方法

  - 步骤：

  - - 1.每个节点不断划分到其他节点的社区使模块度最大化
    - 2.将已划分社区再重复以上思想进行聚类
    - 备注：类似于层次聚类

- HANP

- - 思想：传播能力随着传播距离增大而减弱

- LFM

- - 思想：定义节点连接紧密程度函数，并使社区内该函数值越大

- SLPA

- - 思想：对LPA的优化算法，保留LPA每个节点的更新记录，并选择出现次数最大的类

- BMLPA

- - 思想：定义归属系数，传播计算节点属于周围邻节点的概率

- COPRA

- - 思想：类似于BMLPA，且每个节点归属于类的概率之和为1

- GN

- 思想：不断删除网络中相对于所有源节点最大的边介数的边
- 边介数betweenness：网络中任意两个节点通过该边的最短路径的数量
- GN算法的步骤如下： 

（1）计算每一条边的边介数； 

（2）删除边界数最大的边； 

（3）重新计算网络中剩下的边的边阶数；

（4）重复(3)和(4)步骤，直到网络中的任一顶点作为一个社区为止。

 

（2）DeepWalk：

论文：DeepWalk: Online Learning of Social Representations（2014）

特点：仅涉及网络结构信息，支持在线学习可扩展，并行计算

目标：获得节点出现的概率分布和学习节点表示

数据集：BlogCatalog/Flickr/YouTube

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/1.png">

相关知识：

1）所有的语言模型中需要输入语料库corpus和词vocabulary，拓展到DeepWalk中，而random walk是corpus，graph中的节点为vocabulary

2）采用distribute representation词向量编码，低维实数向量，通常为50维/100维，其最大贡献是相关或者相似词在距离上更近，其距离可以使用余弦相似度或欧氏距离，其通常被称呼为word representation / word embedding.

3）对应one-hot具有维度过大的问题，词的维度即语料库大小

4）Skip-Gram Model是根据某个词分别计算它前后出现某几个词的各个概率。[与其对应的是CBOW模型∈word2vec]

5）Hierarchical Softmax用Huffman编码构造二叉树，其实借助了分类问题中，使用一连串二分类近似多分类的思想。

6）作者不采用CBOW的原因认为基于前后N个词去表示单词的计算量较大

CBOW & Skip-Gram模型 ，而采用SkipGram

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/2.png">

内容：基于NLP中word2vec思想运用到网络的节点表示中，对图中节点进行一串随机游走应用到网络的表示(随机游走序列)，节点类似于词向量，节点间游走类似于句子，从截断的随机游走序列中得到网络的局部信息，再通过局部信息来学习节点的潜在表示。

模型流程：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/3.png">

算法：

1-使用Random walk随机游走生成节点序列

2-遍历所有的节点（SkipGram / Hierarchical Softmax（to speed the training time）/ Optimization(SGD and BP)）

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/4.png">

其中t表示长度；γ为迭代次数；

基于以下公式更新SkipGram算法：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/5.png">

SkipGram:其目的最大化出现在上下文的所有单词的概率以更新向量表示

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/6.png">

其中Φ(vj)是对vj的向量表示

3-为了加速模型训练效率，使用Hierarchical Softmax，在计算Pr(uk|Φ(vj))需要遍历语料库统计uk和vj同时出现的概率，而语料库极大，使得时间复杂度极高，因此使用二叉树进行优化（Huffman树），每个节点在随机游走路径中出现次数作为权重，其根节点是输入节点，非叶子节点为隐层权重，所有的隐层权重及叶子节就是我们需要更新的参数。

其输入vj，Pr(uk|Φ(vj))就是最大化从vj到uk所有路径相乘的概率，二叉树的每一个分支均为二分类问题。

代码：https://github.com/phanein/deepwalk

（3）谱聚类（Spectral Clustering）：

论文：Parallel Spectral Clustering in Distributed Systems（IEEE 2010）

特点：基于图论的聚类方法，仅涉及网络结构信息

内容：利用邻接矩阵A和度矩阵D生成拉普拉斯矩阵L并归一化，计算矩阵L的特征向量和特征值

本质：通过拉普拉斯矩阵变换得到其特征向量组成的新矩阵的 K-Means 聚类，而其中的拉普拉斯矩阵变换可以被简单地看作是降维的过程。而谱聚类中的「谱」其实就是矩阵中全部特征向量的总称。

举例说明：

第一步：构建邻接矩阵A（相连为1，不相连为0）

第二步：构建度矩阵D

第三步构建拉普拉斯矩阵L，L=D-A

备注：拉普拉斯矩阵的构造是为了将数据之间的关系反映到矩阵中

第四步归一化矩阵L即：

第五步计算矩阵L的特征值和特征向量(谱分解)

备注：计算特征值以及特征向量从而达到将维度从N维降到k维

第六步即可采用如K-means聚类

总结：我们要将N维的矩阵压缩到k维矩阵，那么就少不了特征值，取前k个特征值进而计算出k个N维向量P(1),P(2),...,P(k).这k个向量组成的矩阵N行k列的矩阵（经过标准化）后每一行的元素作为k维欧式空间的一个数据，将这N个数据使用k-means或者其他传统的聚类方法聚类，最后将这个聚类结果映射到原始数据中（ 对矩阵N*k按每行为一个数据点，进行k-means聚类，第i行所属的簇就是原来第i个样本所属的簇），原始数据从而达到了聚类的效果。

备注：计算特征值和特征向量

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/7.png">

谱聚类算法的主要优点有：

1）谱聚类只需要数据之间的相似度矩阵，因此对于处理稀疏数据的聚类很有效。这点传统聚类算法比如K-Means很难做到

2）由于使用了降维，因此在处理高维数据聚类时的复杂度比传统聚类算法好。

谱聚类算法的主要缺点有：

1）如果最终聚类的维度非常高，则由于降维的幅度不够，谱聚类的运行速度和最后的聚类效果均不好。

2) 聚类效果依赖于相似矩阵，不同的相似矩阵得到的最终聚类效果可能很不同。

参考：https://blog.csdn.net/songbinxu/article/details/80838865

<https://blog.csdn.net/qq_24519677/article/details/82291867>

<https://www.cnblogs.com/Leo_wl/p/3156049.html>

调用API：sklearn.cluster.SpectralClustering

参考：https://www.jianshu.com/p/d35aea90ec5d

**2.图神经网络的无监督学习算法：**

（1）LINE

代码：https://github.com/tangjianpku/LINE

内容：提出算法定义了两种相似度：一阶相似度和二阶相似度，一阶相似度为直接相连的节点之间的相似性，二阶相似度为共享节点的节点间相似性。

目标：对网络节点进行表达，如n维向量表示某节点，最小化相邻节点对应向量之间的距离，同时最大化不相邻节点对应向量之间的距离。

论文举例：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/8.png">

一阶相似：

1阶相似度用于描述图中成对顶点之间的局部相似度，形式化描述为若u,v之间存在直连边，则边权wuv即为两个顶点的相似度，若不存在直连边，则1阶相似度为0。

如上图，6和7之间存在直连边，且边权较大，则认为两者相似且1阶相似度较高，而5和6之间不存在直连边，则两者间1阶相似度为0。

二阶相似：

虽然5和6之间不存在直连边，但是他们有很多相同的邻居顶点(1,2,3,4)，这其实也可以表明5和6是相似的，而2阶相似度就是用来描述这种关系的。

形式化定义为令pu=(wu,1,...,wu,∣V∣)表示顶点u与所有其他顶点间的1阶相似度，则uu与vv的2阶相似度可以通过pu和pv的相似度表示。若u与v之间不存在相同的邻居顶点，则2阶相似度为0。

参考：https://www.jianshu.com/p/79bd2ca90376

（2） LargeVis

是一种基于LINE的高维数据可视化算法

1、将高维数据表示为网络

2、使用LINE算法将网络嵌入到2维或3维空间

3、绘制出嵌入后得到的向量。

 

（3）图自编码器（Graph Auto-encoder）& VGAE （Variational Graph Auto-Encoders）

目标：链路预测

代码：https://github.com/tkipf/gae

论文：Variational Graph Auto-Encoders

特点：结合了节点特征以及网络结构信息

内容：模型分为2个子模型：（1）推理模型Inference model （2）生成模型Generative model  

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/9.png">

推理模型Inference model ---- 2层GCN

①利用邻接矩阵A(N×N)和度矩阵D(N×N)计算归一化拉普拉斯矩阵A~ 

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/10.png">

②利用特征矩阵X(N×D)和归一化拉普拉斯矩阵A~ 计算GCN(X,A)

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/11.png">

③计算均值向量矩阵µ

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/12.png">

④计算相似对数σ

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/13.png">

⑤随机生成潜在变量矩阵Z(N×F)并更新

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/14.png">

（2）生成模型Generative model

①基于潜在变量矩阵Z更新邻接矩阵A

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/15.png">

备注：σ(·)是logistic sigmoid 函数

（3）Non-probabilistic graph auto-encoder (GAE) model

①VGAE：基于新的邻接矩阵A和特征矩阵X生成新的潜在变量矩阵Z（Embedding）

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/16.png">

参数优化：阅读文献Learning部分

数据集：https://linqs.soe.ucsc.edu/data

（3）反正则化图自编码器Adversarially Regularized Graph Autoencoder for Graph Embedding（ARGA）

& 反正则化变分自编码器 （ARVGA）

论文：Adversarially Regularized Graph Autoencoder for Graph Embedding

代码：https://github.com/Ruiqi-Hu/ARGA

内容：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/17.png">

目的：基于嵌入矩阵Z和节点信息矩阵X(node content matrix)---重构--->图A

谱卷积：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/18.png">

具体f函数：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/19.png">

其中：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/20.png">

Graph Encoder 2-versions:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/21.png">

Graph convolutional encoder:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/22.png">

Variational Graph Encoder:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/23.png">

Link prediction Layer:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/24.png">

Graph Autoencoder layer:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/25.png">

Adversarial Model cost：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/26.png">

Adversarial Graph Autoencoder Model

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/27.png">

Baseline：

1. K-means is a classical method and also the foundation of many clustering algorithms.

2. Graph Encoder [Tian et al., 2014] learns graph embedding for spectral graph clustering.

3. DNGR Deep Neural Networks for Learning Graph Representations (DNGR) [Cao et al., 2016] trains a stacked denoising autoencoder for graph embedding.

4. RTM [Chang and Blei, 2009] learns the topic distributions of each document from both text and citation.

5. RMSC [Xia et al., 2014] employs a multi-view learning approach for graph clustering.

6. TADW [Yang et al., 2015] applies matrix factorization for network representation learning.

 
