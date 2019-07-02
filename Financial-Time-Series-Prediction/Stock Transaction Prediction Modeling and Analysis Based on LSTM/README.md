## Stock Transaction Prediction Modeling and Analysis Based on LSTM


作者：Siyuan Liu

年份：2018

出版：IEEE

结论:

1.新的计算特征[OC OH OL CH CL LH]

2.计算MA & EMA

3.LSTM模型预测

方法流程：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Transaction%20Prediction%20Modeling%20and%20Analysis%20Based%20on%20LSTM/images/1.png">

其中：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Transaction%20Prediction%20Modeling%20and%20Analysis%20Based%20on%20LSTM/images/2.png">

细节：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Transaction%20Prediction%20Modeling%20and%20Analysis%20Based%20on%20LSTM/images/3.png">

A.数据源(CSI 603899 Index from 2014-05-18 to 2017-01-29,2016-12-26 to 2017-01-29为测试集)

B.数据预处理

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Transaction%20Prediction%20Modeling%20and%20Analysis%20Based%20on%20LSTM/images/4.png">

移动平均MA-Moving Average

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Transaction%20Prediction%20Modeling%20and%20Analysis%20Based%20on%20LSTM/images/5.png">

指数移动平均EMA-Exponential Moving Average

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Transaction%20Prediction%20Modeling%20and%20Analysis%20Based%20on%20LSTM/images/6.png">

X是变量,N是某一天,Y是EMA的周期。
从百度wiki中深入了解EMA

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Transaction%20Prediction%20Modeling%20and%20Analysis%20Based%20on%20LSTM/images/7.png">

平滑系数α通常设置为2 / (N + 1), 在EMA计算MACD时，N选择12或26日，则α为2/13或2/27。
特征列表：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Stock%20Transaction%20Prediction%20Modeling%20and%20Analysis%20Based%20on%20LSTM/images/8.png">

C.LSTM模型
