作者：Tanapon Tantisripreecha 
年份：2018
出版：IEEE
动机：预测股价趋势
数据集：APPLE, FACBOOK GOOGLE, AMAZON 

结论:在线学习方法值得尝试

方法流程：
A.  LDA在线学习 （Linear Discriminant Analysis）
在本文中，我们提出了一种将批量学习算法转化为在线学习的LDA-online学习方法。学习框架（如图1）所示，其中T为训练集，N为不可见数据，D为天数序列。LDA在线学习的概念是通过积累新的可用数据来重新考虑训练集，以适应下一个LDA模型
注意，本文提出的方法与传统的在线学习方法不同，传统在线学习方法只考虑预测日期前的固定天数（如图2）。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/1.png">

B. 线性判别分析Linear Discriminant Analysis (LDA) 
LDA的主要目的是基于区间变量的线性组合来预测值。费雪线性判别式（Fisher linear discriminant）是一种寻找特征[5]的线性组合的方法。LDA与统计技术即方差分析（ANOVA）和回归分析密切相关，后者也试图将一个因变量表示为其他特征或测量值[1][2]的线性组合。方差分析的关键概念是使用假设为独立和连续因变量的分类变量，而LDA应用于连续自变量和分类因变量(即类标签)[3]。与方差分析相比，逻辑回归相比方差分析更接近于LDA，因为它们用连续自变量的值来解释分类变量。这些方法适用于自变量为非正态分布的应用，这是LDA方法的一个基本假设是样本是正态分布。

相关资料：

PCA主成分分析是一种无监督的数据降维方法，降维的目标：将数据投影到方差最大的几个相互正交的方向上，以期待保留最多的样本信息，样本的方差越大表示样本的多样性越好，在训练模型的时候，我们当然希望数据的差别越大越好。否则即使样本很多但是他们彼此相似或者相同，提供的样本信息将相同，相当于只有很少的样本提供信息是有用的。样本信息不足将导致模型性能不够理想。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/2.png">

如果同样选择使用PCA，选择方差最大的方向作为投影方向，来对数据进行降维。那么PCA选出的最佳投影方向，将是图中红色直线所示的方向，原因是PCA通常只能处理高斯分布的数据，而图中数据为双高斯分布。

LDA 线性判别分析是一种有监督的数据降维方法，进行数据降维的时候是利用数据的类别标签提供的信息的，线性就是，我们要将数据点投影到直线上（可能是多条直线），直线的函数解析式又称为线性函数。

C. 批量学习模型
批量学习算法从训练集中学习建立模型，并将其模型应用于预测新的未知数据。批量学习对于时间序列数据域是不实用的，因为概念模型是静态的，而不可见数据的值取决于时间的函数。我们发现当前数据的值影响下一个值。考虑股票预测领域，我们发现批量学习通常使用简单移动平均(SMA)、指数移动平均(EMA)和相对奇异指数(RSI)来计算历史股票价格(见公式1-3)。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/3.png">

注意：AvgU是最近n个价格条中所有向上移动的平均值，AvgD是最近n个价格条中所有向下移动的平均值，n是RSI的周期。
SMA、EMA和RSI是对投资者有用的知名指标。移动平均线(SMA)和均线(EMA)是投资者通常用来考虑趋势的基本估计工具的平均价格运动，RSI（Relative Strange Index）是投资者用来检测股票趋势的指数。

A.数据集
我们从雅虎财经收集纳斯达克每日股票市场 <https://finance.yahoo.com/>。表中显示了四个受欢迎的IT公司股票的数据集，这些股票的交易量相当大。这些股票包括苹果(Apple, 19992017)、Facebook (Facebook, 2012-2017)、谷歌(GOOGL, 20042017)和亚马逊(Amazon, 1999-2017)。股票信息包括日期、开盘价、最高价、最低价和收盘价。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/4.png">

B.预测方法
LDA-Online的目标分类是为了预测股票的走势。为了考虑股票的走势，将当前的预测价格与之前的股票价格进行比较，从而确定(上涨或下跌)的方向。如果预测的价格高于之前的股票价格，那么移动就是向上的。如果当前股价低于之前的股价，则判定为下跌。我们评估了所有算法的性能，考虑股票的方向和衡量方面的准确性。

[1] J. Patel, S. Shah, P. Thakkar, and K. Kotecha, “Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques”, Expert Systems with Applications, vol. 42, p. 259-268, 2015.

[2] M. Ballings, D. Van den Poel, N. Hespeels, and R. Gryp, “Evaluating multiple classifiers for stock price direction prediction”, Expert Systems with Applications, vol. 42, no.20, p. 7046-7056, 2015.

[3] M. Amin Hedayati, M. Moein Hedayati, E.Morteza, “Stock market index prediction using artificial neural network”, Journal of Economics, Finance and Administrative Science, p. 89-93, 2016.

[5] J. Patel, S. Shah, P. Thakkar, and K. Kotecha, “Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques”, Expert Systems with Applications, vol. 42, no 1, p. 259-268, 2015

补充：（李政轩的LDA教学视频）

LDA：类间距离越大越好；类内距离越小越好；

1.xj与v的内积（统计学）

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/5.png">

2.计算xj投影在v方向上的坐标值为<xj,v>（有正有负，在0点左右），其中v是主轴方向，内积乘以主轴方向就是生成新的向量

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/6.png">

符号表示说明：
	L:种类数
	Ni:类别i的样本数
	N:全部样本数
	X(i)j:第i类的第j个样本
1.X(i)j样本投影到方向v上的坐标为vTx(i)j

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/7.png">

2.计算投影后的均值

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/8.png">

3.使类间均值越大

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/9.png">

其中，类间均值差距之和越大，其中因为每类的样本数不同而添加权重Ni/N，且mi[mean]-mj[mean]均为数值，因此其数值差的平方等于本身乘以其转置。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/10.png">

分析SbLDA

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/11.png">

4.计算类内距离

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/12.png">

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Market%20Movement%20Prediction%20using%20LDA-Online%20Learning%20Model/images/13.png">

