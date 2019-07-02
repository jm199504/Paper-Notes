## Applied attention-based LSTM neural networks in stock prediction
作者：Li-Chen 

年份：2018

出版： IEEE

结论：加入了注意力机制的LSTM预测股价

动机：虽然注意力机制最近在神经机器翻译中得到了广泛的应用，但基于注意力的股票预测深度学习模型的研究却很少

图1

A. 数据集介绍
B. 深度学习模型 
在LSTM神经网络模型中加入注意机制，是提高神经网络能力和可解释性的有效途径。 
我们模型的输入是一系列股票数据，包括价格数据和技术指标，模型为多分类输出（使用softmax函数），其中
	• Class 0 represents a stock price increase of more than 3%, 
	• Class 1 an increase of 2% to 3%, 
	• Class 2 an increase of 1% to 2%, 
	• Class 3 an increase of 0% to 1%, 
	• Class 4 a flat stock price, 
	• Class 5 a stock price decrease of 0% to 1%, 
	• Class 6 a decrease of 1% to 2%, 
	• Class 7 a decrease of 2% to 3%, 
	• Class 8 a decrease of more than 3%.

C. 交易模型 
我们使用了来自深度学习模块的预测结果。当预测类属于递增类时——也就是说，我们的模型预测股票价格将会上涨——策略是购买股票。当预测股票价格将下跌时，策略就是卖出股票。计算结果与实测结果拟合较好
D.结果评估