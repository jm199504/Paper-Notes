## A Quick Gbest Guided Artiﬁcial Bee Colony Algorithm for Stock Market Prices Prediction

作者：Habib Shah 

年份：2018

出版：Symmetry 2018

目的：股市趋势预测

数据集：Saudi Stock Market (SSM)

贡献：提出了A Quick Gbest Guided artificial bee colony （QGGABC-ANN）

总结：了解ABC算法可以实现参数寻优，本文优化ABC算法达到更高的准确性

方法：

1.多层感知机Multilayer Perceptron (MLP)：1层输入层；2层隐藏层；1层输出层

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/A%20Quick%20Gbest%20Guided%20Arti%EF%AC%81cial%20Bee%20Colony%20Algorithm%20for%20Stock%20Market%20Prices%20Prediction/images/1.png">

x是输入向量；b是偏置；f是激活函数，计算公式如下：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/A%20Quick%20Gbest%20Guided%20Arti%EF%AC%81cial%20Bee%20Colony%20Algorithm%20for%20Stock%20Market%20Prices%20Prediction/images/2.png">

最常见的激活函数是sigmoid函数：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/A%20Quick%20Gbest%20Guided%20Arti%EF%AC%81cial%20Bee%20Colony%20Algorithm%20for%20Stock%20Market%20Prices%20Prediction/images/3.png">

传统算法陷入局部最小值的方法由于错误的参数选择，不合适的网络结构，完全随机的方法训练(原因)。为了克服它,仿生学习算法基于生物启发的学习算法，提出了ABC（人工蜂群算法，Artificial bee conlony），ACO，CS算法。

人工蜂群算法Artificial Bee Colony Algorithm

根据蜜蜂的特性分为受雇蜂、旁观蜂和侦察蜂，利用过程由受雇蜂和旁观蜂进行，侦察蜂利用策略进行探测过程，如下式所示。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/A%20Quick%20Gbest%20Guided%20Arti%EF%AC%81cial%20Bee%20Colony%20Algorithm%20for%20Stock%20Market%20Prices%20Prediction/images/4.png">

其中，Vij表示受雇蜂在xij附近的新解决方案的数量，k是i附近的新蜜源，θ是[-1,1]间的随机数，寻找新蜜源：其中xijmin表示最小值，随机寻找最大值与最小值间的值。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/A%20Quick%20Gbest%20Guided%20Arti%EF%AC%81cial%20Bee%20Colony%20Algorithm%20for%20Stock%20Market%20Prices%20Prediction/images/5.png">

1.1雇佣型蜜蜂和蜜源一一对应，根据Vij=xij+θ(xij-xkj)更新蜜源信息和确定蜜源数量

1.2旁观蜂根据雇佣蜂所提供的信息采用一定的选择策略(或依据概率，采用轮盘赌)选择蜜源，更新蜜源信息，同时确定蜜源的花蜜量

1.3确定侦察蜂，并根据xid=xdmin+rand(0,1)(xjimax-xsjmin)寻找新蜜源

1.4记忆目前为止找到的最好的蜜源

总结：

雇佣蜂：利用先前的蜜源信息寻找新的蜜源并与观察蜂分享蜜源信息；

观察蜂：在蜂房中等待并依据雇佣蜂分享的信息寻找新的蜜源；

侦查蜂：寻找一个新的有价值的蜜源，它们在蜂房附近随机的寻找蜜源。

2.ABC算法仍然会陷入局部最优，因此本文提出了Quick Gbest Guided Artiﬁcial Bee Colony Algorithm（优化传统的人工蜂群寻优算法，因此该文创新点在于改进寻优算法，应用于股票预测）

知识点：

1.特征选择方法:

	1.1 Principal Component Analysis（PCA）
	
	1.2 GA( genetic algorithms)
	
	1.3 Decision trees with BP algorithm
	
2.感知器无法解决非线性XOR分类问题

2.1 XOR表示异或，异或是线性不可分的

2.2 异或问题：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/A%20Quick%20Gbest%20Guided%20Arti%EF%AC%81cial%20Bee%20Colony%20Algorithm%20for%20Stock%20Market%20Prices%20Prediction/images/6.png" width="200">

即：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/A%20Quick%20Gbest%20Guided%20Arti%EF%AC%81cial%20Bee%20Colony%20Algorithm%20for%20Stock%20Market%20Prices%20Prediction/images/7.png" width="200">

感知机可表示为超平面画一条分界线使其分类，因此无法分割。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/A%20Quick%20Gbest%20Guided%20Arti%EF%AC%81cial%20Bee%20Colony%20Algorithm%20for%20Stock%20Market%20Prices%20Prediction/images/8.png" width="200">

3.FFNN:Feed forward neural networks(前向反馈网络，神经元间完全相连）

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/A%20Quick%20Gbest%20Guided%20Arti%EF%AC%81cial%20Bee%20Colony%20Algorithm%20for%20Stock%20Market%20Prices%20Prediction/images/9.png">
