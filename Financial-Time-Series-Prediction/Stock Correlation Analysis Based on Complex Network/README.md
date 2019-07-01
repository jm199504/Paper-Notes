## Stock Correlation Analysis Based on Complex Network
作者：Shutian Li
年份：2016
出版：IEEE

结论：分析两种股票对应窗口的相关系数，计算相似度。
目的：挖掘SSE对应公司(Shanghai Stock Exchange)与SSE指数(SSE complex index)间关系

方法：
A.数据集
The Shanghai Composite Index [上海综合指数]
B定义参数δT 
我们将这些数据分成M个窗口[10]，以t=1,2，…宽度为T的M，即日收益率为M（将数据划分为M个窗口对应T为期窗口大小）。

图1

滑动窗口会导致窗口间有重叠，其不重叠部分长度为δT
C.定义相关系数
基本数据集包含N个资产，收盘价表示为Pi(τ)对于第i个资产于第τ日，对数收益公式如下：

图2

为了量化资产i和j在t时刻的相似程度，我们将i和j在t时刻窗口的相关系数定义为对数收益，即

图3

D. 选择参数T和δT 

窗宽T的选取是一种权衡，当T过小时数据存在很大的噪声，当T较大时数据过于平滑。Onnela提出当窗口步长T=21天和T=1000时最优。 

 

参考文献：

[12] Tumminello M, Lillo F, Mantegna R N. Correlation, hierarchies, and networks in financial markets[J]. Journal of Economic Behavior & Organization, 2010, 75(1): 40-58. 

[13] Keskin M, Deviren B, Kocakaplan Y. Topology of the correlation networks among major currencies using hierarchical structure methods[J]. Physica A: Statistical Mechanics and its Applications, 2011, 390(4): 719-730. 

[14] Newman M E J. Mixing patterns in networks[J]. Physical Review E, 2003, 67(2): 026126

 

但并不适合我们的数据集，因此本文选取T=200, 100，窗步长T=21天（固定在一个月的时间内），分析其对应的平均相关系数的概率密度，定义为

图4

E. 确定关系矩阵 

F. 分析实验结果 

G. 基于k -均值算法的网络社区结构分析 

图5

结论：

股票之间的相互作用在复杂指数上升时减弱，在指数下降时增强。同时，节点的分布在繁荣时期是稀疏的，在衰退时期是密集的。

 

通过对聚类结果的分析，我们发现，正如我们所预料的那样，同一社区的节点基本上处于相同或相似的行业。

 

在时间窗t=10时，节点间相关性较大，从图可以看出，群落较为密集，最大的群落有37个节点，最小的群落有5个节点。

 

社区分布相对平均，有7个社区成员超过10人， 这是因为在时间窗t=19时，节点之间的相关性在市场繁荣时期较小，高涨时期股票聚类时各簇的数量较为平均，即不出现数量较大的簇。

 

当复杂指数上升时，股票之间的相互作用减弱，当指数下降时，股票之间的相互作用增强，其依据是市场鼎盛时期即指数价格高涨时股票间聚类时，最大簇的股票数量较小，即不均衡；而市场疲软时期即指数价格低迷股票聚类时，最大簇的股票数量较大，而最小簇里的股票数量较小，分散均衡/平均。