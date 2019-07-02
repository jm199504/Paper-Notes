## Stocks Analysis and Prediction Using Big Data Analytics

作者：ZhihaoPENG

年份：2019

出版：2019 International Conference on Intelligent Transportation, Big Data & Smart City (ICITBS)

动机：预测未来原油价格 

数据集：The United State Oil Fund,USO

结论:主要介绍hadoop和spark在大数据框架上的应用。

方法流程：

A. USO数据集介绍

美国石油基金USO（ the United State Oil Fund）是一种交易所交易基金(exchange-traded fund ，ETF)证券，旨在跟踪西德克萨斯中质油和低硫原油的每日价格波动。USO的目标是监测其份额百分比的每日变化。由于USO追踪的是美国纽约商业交易所（New York Mercantile Exchange）上市的近一个月合约（contracts listed），这些数据可以用来理解原油市场的短期波动。USO的每日收盘价可以在雅虎财经网站（Yahoo Finance website）上找到。

B. Hadoop框架介绍

Apache Hadoop是一个开源的大数据框架，为通过分布式存储和处理处理大型数据集提供了一个平台。该框架基于以下假设:硬件故障是常见的，因此其设计能够自动处理所有可能的系统故障（个人见解：同一切块文件存储于多个DataNode）。生态系统的核心是Hadoop兼容的文件系统(HDFS)和MapReduce，因此它经常被称为Hadoop MapReduce框架[2]。这个框架是展示分布式和并行计算能力的完美例子。处理大量数据的应用程序可以很容易地在Hadoop MapReduce框架上编写。数据在Hadoop生态系统上的多台机器(分布式架构)上并行处理(并行计算)。简而言之，目标是将任务分布到多个集群中，这些集群具有一个可兼容的文件系统。Hadoop生态系统的核心是Hadoop分布式文件系统(HDFS)和MapReduce(如图1所示)。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stocks%20Analysis%20and%20Prediction%20Using%20Big%20Data%20Analytics/images/1.png">

C. 在Spark上实现机器学习（Spark介绍）
Spark是Hadoop生态系统中的一个领先工具。Hadoop中的MapReduce只能用于批处理，不能处理实时数据。Spark可以独立工作，也可以通过Hadoop框架来利用大数据，并在分布式计算环境中执行实时数据分析。它可以支持所有类型的复杂分析，包括机器学习、商业智能、流处理和批处理。Spark在大规模数据处理方面的速度是Hadoop MapReduce框架的100倍，因为它在内存中执行计算，从而提供了Mapreduce增速。大数据时代不仅迫使我们考虑能够快速存储和处理数据的框架，还要求我们考虑实现机器学习(ML)算法的平台，这种算法在许多领域都有应用。幸运的是，Spark提供了一个灵活的平台来实现许多机器学习任务，包括分类、回归、优化、聚类、维数、化简等。

流程线路：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stocks%20Analysis%20and%20Prediction%20Using%20Big%20Data%20Analytics/images/2.png">

A. 数据特征

我们的数据集包括13个石油股票从SP500股票可用的雅虎金融（包含2905行和13列）

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stocks%20Analysis%20and%20Prediction%20Using%20Big%20Data%20Analytics/images/3.png">

B. 数据注入（Data Injection）

数据通过Flume注入HDFS，三个组件协同工作，将数据放入Flume，这三个组件分别是是 Source(源), Channel(通道)，Sink(槽)。在shell终端上执行命令访问Source(源)，本地内存充当Channel(通道)，DFS(分布式文件系统Distributed File System)是Sink(槽)。数据注入需要配置本地数据库为HDFS。

C. 数据存储

D. 数据预处理

E. 线性回归

参考文献：
[2] White, T. (2011). Hadoop: the definitive guide. Sebastopol, CA: OReilly
