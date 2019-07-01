作者：Habib Shah 
年份：2018
出版：Symmetry 2018
目的：股市趋势预测
数据集：Saudi Stock Market (SSM)
贡献：提出了a Quick Gbest Guided artificial bee colony （QGGABC-ANN）
总结：了解ABC算法可以实现参数寻优，本文优化ABC算法达到更高的准确性

方法：
1.Multilayer Perceptron (MLP):The basic architecture of MLP with one input, two hidden and one output layers is shown in Figure 1.

图1

Weights vector to be x; x is the vector of inputs; b is the bias, and f is the activation function then output through MLP neurons is computed, mathenmatically, as

图2

The most commonly-used hidden neuron activation function is sigmoid function which is given by equation:

图3

The conventional algorithms methods trapped in local minima[缺陷] due to suboptimal weight values, wrong number of parameters selection, unsuitable network structure, and the fully random method of training[原因] as well. In order to overcome it, bio-inspired learning algorithms[解决办法:基于生物启发的学习算法] have been proposed for MLP such as ABC, ACO, and CS[ABC:Artifical bee conlony]. 

Artificial Bee Colony Algorithm ABC[人工蜂群算法]

The characteristics of bees are divided into three aspects namely employed, onlooker, and scout bees[蜜蜂分工:雇佣型、旁观型和侦察型]. During the process, exploitation process is carried out by the employed and onlooker bees [利用过程:雇佣和旁观型蜜蜂]while scout bees are used for the exploration process[探索过程:侦察型蜜蜂] through the following strategy, as given in Equation. 

图4

where Vij represents the number of new solutions in the neighbourhood of xij for the employed bees[公式一：更新蜜源信息和确认蜜源的花蜜数，其中Vij表示目前最优参数], k is a solution in the neighbourhood of i[k是i附近的新蜜源], and θ is a random number in the range [−1, 1]. [公式二：寻找新蜜源,其中xijmin表示其最小值，随机寻找最大值与最小值之间的值]

图5

1.1雇佣型蜜蜂和蜜源一一对应，根据Vij=xij+θ(xij-xkj)更新蜜源信息和确定蜜源数量
1.2旁观蜂根据雇佣蜂所提供的信息采用一定的选择策略(或依据概率，采用轮盘赌)选择蜜源，更新蜜源信息，同时确定蜜源的花蜜量
1.3确定侦察蜂，并根据xid=xdmin+rand(0,1)(xjimax-xsjmin)寻找新蜜源
1.4记忆目前为止找到的最好的蜜源

总结：

雇佣蜂：利用先前的蜜源信息寻找新的蜜源并与观察蜂分享蜜源信息；
观察蜂：在蜂房中等待并依据雇佣蜂分享的信息寻找新的蜜源；
侦查蜂：寻找一个新的有价值的蜜源，它们在蜂房附近随机的寻找蜜源。

2.ABC算法仍然会陷入局部最优，因此本文提出了Quick Gbest Guided Artiﬁcial Bee Colony Algorithm[优化传统的人工蜂群寻优算法，因此该文创新点在于改进寻优算法，应用于股票预测]

知识点：

1.特征选择方法:
	1.1 Principal Component Analysis
	1.2 GA( genetic algorithms)
	1.3 decision trees with BP algorithm
2.Perceptron cannot solve the non-linear XOR classification problems
2.1 XOR表示异或，异或是线性不可分的
2.2 异或问题：

图6

即：

图7

感知机可表示为超平面画一条分界线使其分类，因此无法分割

图8

3.FFNN:Feed forward neural networks(前向反馈网络)[神经元间完全相连]

图9

