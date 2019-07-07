作者：Heng-Tze Cheng等

应用领域：推荐系统

调研学习：

1.推荐系统主要分为两个部分：
	
检索系统(Retrieval)和排序系统(Ranking)。
	
2.memorization & generalization：
	
推荐系统重要问题之一，解决memorization（记忆）和generalization（归纳/泛化）。
	
memorization主要是基于历史数据学习频繁共同出现的item，并且探索他们之间的相关性，由wide作为主导；
	
generalization主要是基于相关性之间的传递， 发现新的特征的组合，由deep作为主导。
	
3.Wide model主要采用逻辑回归，且特征一般为分类值(categorical)，通常二值且稀疏，用one-hot编码（对于连续值可以考虑使用Bucketization（桶化），将连续值分组看待）；Deep model采用神经网络，特征为连续值(continuous)，通常会归一化到[0,1]，激活函数通常为ReLu。
	
摘要：具有非线性特征转换的广义线性模型被应用于具有稀疏输入的回归/分类问题， 当特征工程较少特征时，深度神经网络通过对稀疏特征的低维稠密嵌入学习组合能较好地挖掘隐含特征。然而，当用户与项目之间的交互是稀疏的、高秩的时，带有嵌入式的深度神经网络不建议使用。

在本文中，我们提出了Wide & Deep Learning 模型，联合线性模型和深度神经网络将记忆和归纳的优点结合，并应用于推荐系统，正如下图所示：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems/images/1.png">

主要贡献：

1.Wide & Deep learning 模型联合前馈神经神经和特征转换的线性模型实现稀疏特征的通用的推荐系统。

2.Wide  &Deep Learning 推荐系统用于GooglePlay移动端应用商店（十亿活跃用户和百万应用）。

3.开源基于TensorFlow的API（详情查看：http://tensorflow.org）

推荐系统流程一览：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems/images/2.png" width="500">

推荐系统具体细节：

用户特征（User features）：country, language, demographics

上下文特征（Contextual features）：device, hour of the day, day of the week

印象特征（Impression features）：app age, historical statistics of an app

模型细节：

1.Wide组件：图1左侧，需要对离散特征进行交叉积变换（cross-product transformation）：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems/images/3.png">

其中 x = [x1; x2; …; xd] 表示含有d个特征的向量，[w1; w2;…; wd]是模型参数，模型通常训练 one-hot 编码后的二值特征，cki是布尔值变量，φk是第k次变换，对于二分类特征，交叉积变换：例如gender=female & language=en为1，其余均为0，该变换可以捕获二元特征间相互作用，为广义线性模型增加了非线性，但是对于在训练集里没有出现过的 query-item pair，该交叉积变换不会归纳出训练集中未出现的特征对。

2.Deep组件：图1中右侧，前馈神经网络，通常会将特征embedding到10-100维，再投入隐藏层中，该Deep用于探索新的特征组合，提高推荐系统的多样性，或提升模型泛化能力，但是query-item matrix 非常稀疏，很难学习，然而 dense embedding 的方法还是可以得到对所有 query-item pair 非零的预测，这就会导致 over-generalize，推荐相关弱的物品：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems/images/4.png">

3.Joint Training Wide & Deep Model：结合Wide和Deep，使用加权求和方式输出预测值，通常再使用逻辑损失(Logistic Loss)映射。(插语：联合学习和集成学习不同，集成学习是独立训练即互不了解，而联合学习是同时训练即模型参数是同时被更新的)，其中Wide模型的少量交叉积可以弥补Deep模型的缺陷（无需全量Wide模型）。

优化器：使用带有L1正则化的Followthe-regularized-leader (FTRL) algorithm优化Wide；使用AdaGrad优化Deep。

训练方式：Mini-batch stochastic optimization

FTRL论文：https://www.semanticscholar.org/paper/7bdf20d18b5a9411d729a0736c6a3a9a4b52bf4f?p2df

AdaGrad论文：《Adaptive Subgradient Methods for Online Learning and Stochastic Optimization∗》

Mini-batch论文：《Efficient Mini-batch Training for Stochastic Optimization》

模型输出公式：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems/images/5.png">

其中Y表示二元标签，σ(·)是sigmoid函数， φ(x) 是交叉积转换（cross product transformations），b是偏置项（bias），W(wide)为Wide模型权重，W(deep)为Deep模型权重，

论文模型具体流程和函数使用：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems/images/6.png" width="700">

具体描述：在Deep模型，使用32维度embedding向量学习每一个分类特征，拼接（concatenate）所有embedding后大约1200维度，再投入3层ReLu层，最后使用logistic loss优化模型参数，5千亿的数据样本，且时刻会有新的训练数据产生，而模型需要再训练，然而再训练是十分消耗计算资源和耽误时间，为解决该问题，Google大佬们实现了热启动系统初始化带有embedding和线性模型权重的新模型（基于之前的模型）。

结论：记忆和归纳对推荐系统都十分重要，其中Wide模型可以通过交叉积特征转换（cross-product feature transformations）有效地记忆稀疏特征的交互，而Deep模型（深度神经网络）可以通过低维嵌入归纳到未可见的特征交互，该模型集成了Wide和Deep模型的优势，通过对大型商业应用Google Play的推荐系统框架进行了应用和评估，在线实验结果表明该模型优于Wide模型和Deep模型。

Wide & Deep 和 Deep & Cross 及tensorflow实现：https://blog.csdn.net/yujianmin1990/article/details/78989099

个人总结：Wide&Deep模型，其中Wide模型是用于记忆，即对于训练集中出现的query-item pair出现过的特征组合进行标记，而对于未出现过query-item pair无法进行标记，因此考虑Deep模型可以用于归纳，即探索新的特征组合（推荐物品的多样性），将高维度的类别特征映射为低维度的向量，其中Wide模型的输入是特征组合的交叉积变换，Deep模型的输入是类别特征(产生embedding)+连续特征，最后对这两部分模型进行加权求和作为最后的推荐得分，再对其进行排序推荐。
