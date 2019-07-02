## Stock price forecast using wavelet ransformations in multiple time windows and neural network

作者：Ajla Kulaglic

年份：2018

出版：IEEE

总结：采用离散小波变换和2个神经网络模型（一层/两层隐藏层）

贡献：将相同的数据集输入到不同的模型中可以比较出更好的模型，并可以对预测结果进行聚类。

目的：股价预测

流程：

A.APPLE股价2008.5-2018.5(2520交易日)

2种不同的数据集：

第一个输入数据集包含8个工作日(星期三和星期五结束，个人认为表示下个星期五)

第二个输入数据集需要4个工作日(星期二到星期五结束)

输出数据集包括5个工作日（每周滑动）

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20price%20forecast%20using%20wavelet%20ransformations%20in%20multiple%20time%20windows%20and%20neural%20network/images/1.png">

B. 方法

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20price%20forecast%20using%20wavelet%20ransformations%20in%20multiple%20time%20windows%20and%20neural%20network/images/2.png">

评估标准：Root Mean Square Error(RMSE)

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20price%20forecast%20using%20wavelet%20ransformations%20in%20multiple%20time%20windows%20and%20neural%20network/images/3.png">
