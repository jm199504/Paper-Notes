作者：Roger Achkar

年份：2018

出版：IEEE

数据集：FacebookTM stocks, GoogleTM stocks, and BitcoinTM stocks

结论：Adam是最佳选择优化器，Sigmoid是LSTM预测模型中的最佳选择激活函数。

方法：

A.激活函数

由于逻辑函数在0上是对称的，这使得训练更加简单和准确.

B. 权重和偏置

权重和偏差最初是随机产生的。每次反向传播时，权值都会相应地进行修正

C. 神经元和层数

MLP分为三个主要层:输入层、输出层和隐藏层。然而，对于网络的大小(即隐藏神经元的数量)并没有明确的规定。根据问题的非线性程度和维数得出平均隐藏神经元。过多的隐藏神经元会导致神经网络的过度学习。另一方面，太少的神经元将不能给网络准确学习的自由。此外，需要至少有一个隐藏层，但是有多个就足够了。一个隐层可能需要太多的隐神经元，因此在实际应用中，两个隐层对非线性问题[10][11][12][13][14]的建模效果更好。经过多次尝试，本文将遵循7-3-2-1神经元的拓扑结构。网络从7个输入神经元开始，七天数据预测第八天。

D. 训练模型（学习率=0.02）

E.LSTM-RNN
LSTM网络是一种特殊的RNN，具有学习长期依赖关系的能力。网络的记忆细胞取代了传统隐藏层中的人工神经元。有了记忆单元，网络能够有效地将记忆关联起来，从而随着时间的推移动态地掌握数据结构，具有较高的预测能力。LSTM体系结构的基本单元是块内存，其中包含一种或多种不同类型的内存单元和三种自适应乘法，分别称为输入门、遗忘门和输出门。
输入和输出门乘以单元格的输入和输出，而遗忘门乘以单元格的前一状态。
门的激活函数通常是逻辑sigmoid函数，所以它介于0(门关闭)和1(门打开)之间。细胞输入或输出的激活函数通常是tanh或logistic sigmoid函数。

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Financial-Time-Series-Prediction/Comparison%20of%20BPA-MLP%20and%20LSTM-RNN%20for%20Stocks%20Prediction/images/1.png">

F. Adam优化算法

Adam代表自适应力矩估计（ Adaptive Moment Estimation）。Adam算法可以代替传统的随机梯度下降法，在训练数据的基础上迭代更新网络权值。该算法的一些特点是:

(1) 直接实现（Straight forward to implement | 并不确定）。

(2) 计算效率高。

(3) 内存需求小。

(4) 梯度的对角缩放具有不变性。

(5) 非常适合处理数据较大/参数较多的问题。

(6) 适用于非平稳目标（Appropriate for non-stationary objectives | 并不确定）。

(7) 适用于非常嘈杂/稀疏梯度的问题。

(8) 超参数有直观的解释，通常需要很少的调优

Adam被证实其收敛性符合理论分析的预期，将Adam应用于多层神经网络的逻辑回归算法中，证明了利用大模型和数据集可以有效地解决实际的深度学习问题。

参考文献：
[10] C. Saide, R. Lengelle, P. Honeine, C. Richar and R. Achkar, "Nonlinear adaptive filtering using kernel-based algorithms with dictionary adaptation. International Journal of Adaptive Control and Signal Processing," 2015.

[11] G. A. Kassam and R. Achkar, LPR CNN Cascade and Adaptive Deskewing.

[12] M. Owayjan, R. Achkar and M. Iskandar, "Face Detection with Expression Recognition using Artificial Neural Networks," Beirut, Lebanon, 3rd Middle East Conference on Biomedical Engineering (MECBME), October 2016, pp. 116-120. 

[13] R. Achkar, M. ElHalabi, E. Bassil, R. Fakhro and M. Khalil, "Voice identity finder using the back propagation algorithm of an artificial neural network," 2016. 

[14] Y. Harkouss, S. Mcheik and R. Achkar., "Accurate wavelet neural network for efficient controlling of an active magnetic bearing system," 2010. 
