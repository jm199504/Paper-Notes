作者：Athit Phongmekin 
年份：2018
出版：IEEE
目的：分析股票趋势

文献回顾
本文利用泰国证券交易所(SET)金融类股的财务比率和公司行业数据，构建了两种预测分类模型
(i) 股票的一年回报率是否会超过既定指数及
(ii) 回报是否为正。各种分类技术，包括
	• 逻辑回归Logistic Regression (LR)
	• 决策树Decision Tree (DT)
	• 线性判别分析Linear Discriminant Analysis (LDA)
	• K近邻K-Nearest Neighbor (KNN) 
通过对ROC曲线下面积(AUC)的分析，对其性能进行了评价。
1 预测股价
2 预测股票的质量表现
作为Hiral等人[2]工作的补充，Alostad和Davulcu[3]将Twitter等社交媒体平台的新闻整合到支持向量机(SVM)和LR中，预测股价走向。结果表明，在社交媒体新闻中使用LR的准确率更高，达到70%以上。

Dutta等人也使用LR来预测某只股票是否会超过印度股市指数(NIFTY)，使用12个月的财务比率作为独立变量。与其他研究不同的是，所提出的模型非常实用，因为我们可以衡量个股对整体市场的回报，这代表了投资于市场的机会成本。在他们的研究中，LR被认为是一个比较好的分类器，其预测精度达到74.6%。

Tsai和Wang[5]利用人工神经网络(ANN)和决策树(DT)分类技术，建立了台湾电子产业[5]股票涨跌的预测模型。本文证明了使用基础、技术和宏观经济参数的ANN-DT混合模型比单独使用ANN或DT具有更高的精度。

在文献综述的基础上，我们提出了对KNN、LDA等较为简单的分类器以及SET金融板块中较为先进的股票分类器的使用进行深入的研究。我们还介绍了使用AUC来测量模型性能而不是准确性，因为AUC可以更好地显示模型的有用性——AUC可以分别调查真实阳性率和假阳性率。

方法流程：
1 自变量 
我们使用四种不同的分类技术来基于历史数据预测股票的表现，其中一个数据点包含33个与金融相关的自变量和10多个与行业/子行业分类相关的变量。根据模型的目标选择了两个因变量，即一年相对于SET指数的相对表现和一年的价格变动。从2000年到2013年，总共有452个数据点。训练数据与测试数据的比例设置为70:30。建立模型后，利用后向消元法和前向选择法进行变量选择，对每种方法的AUCs进行优化。将分类模型应用于测试数据集后，通过绘制不同阈值下的真阳性率与假阳性率，构建计算AUCs的ROC曲线。本文利用Rapidminer数据挖掘软件对AUC进行了计算。

由于我们使用11个不同财务比率和10个行业分类的3个年终值作为自变量，那么每个模型中的自变量数量就等于43。虽然财务比率的选择选择基于之前的研究和他们的可用性[6]——包括Priceto-Earnings比率(PE),市净率(PB)比,股本回报率(ROE),投资资本回报率(ROIC),资产回报率(ROA),资产周转率,收入增长,净利润增长,净债务与股本比率,利润率,股息收益率——由于没有最著名的分类标准能够完美地描述公司的业务和特征，因此采用了各种行业分类变量。更正式地说，(子)行业是按照彭博行业分类体系(BICS)一级行业、BICS二级行业集团、全球行业分类标准(GICS)行业、GICS行业集团、GICS子行业、行业分类基准(ICB)行业、ICB子行业、行业集团、行业发行机构、行业指数名称进行分类的。
2 因变量
计算股票和集合的收益(泰国证券交易所)

图1

3 分类方法 
3.1 CART决策树
分类回归树(CART)是数据分类中常用的决策树之一。与Chisquare自动交互检测器(CHAID)决策树不同，CART使用基尼指数或信息熵来实现这一目的。基尼指数(Gini index)是统计学家科拉多•基尼(Corrado Gini)在1912年提出的一种统计离散度指标，主要用于衡量经济不平等。信息熵是由给定变量的负对数概率质量函数之和定义的另一种度量。这一措施是由克劳德·香农于1948年提出的，当时使用的单位是比特。通常，高度不确定性的事件会产生更高的信息熵，因为通过更多的随机抽样可以获得更多的信息。但是，与此相反，没有不确定性的事件不会提供新的信息，并且信息熵为零。
3.2  逻辑回归Logistic Regression (LR) 
3.3  线性判别分析Linear Discriminant Analysis (LDA) 
LDA是Ronald Fischer在1936年提出的一种参数分类方法，因变量是分类的，自变量是连续的。LDA的基本概念是得到一个线性回归方程，该方程最好地分离了这两种分类。使用LDA的缺点是，LDA需要严格的正态性和同方差假设——尽管Mircea等人[8]认为违反正态性假设不是致命的，并且显著性检验仍然是可信的。 
3.4  K-Nearest Neighbor 
K近邻(KNN)是一种非参数分类器，其中，给定一组测试数据，KNN根据(i)距离函数和(ii) K个最近的距离按升序分为两组，即K个最近的数据点。各种距离函数可以用来计算数据点之间的距离，例如曼哈顿（Manhattan）距离和欧式（Euclidean）距离。

4.数据标准化

实验结果：

图2

[2] Hiral R. Patel, Satyen M. Parikh, and D. N. Darji, "Prediction model for stock market using news based different Classification, Regression and Statistical Techniques: (PMSMN)," presented at the 2016 International Conference on ICT in Business Industry & Government (ICTBIG) Indore, India, 2016.  
[3] H. Alostad and H. Davulcu, "Directional prediction of stock prices using breaking news on Twitter," Web Intelligence, vol. 15, no. 1, pp. 1-17, 2017. 
[4] Avijan Dutta, Gautam Bandopadhyay, and S. Sengupta, "Prediction of Stock Performance in the Indian Stock Market Using Logistic Regression," International Journal of Business and Information, vol. 7, no. 1, pp. 105-136, 2012. 
[5] C.-F. Tsai and S.-P. Wang, "Stock Price Forecasting by Hybrid Machine Learning Techniques," in International MultiConference of Engineers and Computer Scientists, Hong Kong, 2009, vol. 1. 
[8] Gabriela Mircea, Marilen Pirtea, Mihaela Neamtu, and S. Bazavan, "Discriminant analysis in a credit scoring model," presented at the Recent Advances in Applied and Biomedical Informatics and Computational Engineering in Systems Applications, Florence, 2011.