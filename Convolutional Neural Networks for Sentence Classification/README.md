Convolutional Neural Networks for Sentence Classification

**双通道模型架构图例  Model architecture with two channels for an example sentence **

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/1.png">

输入层：

接收一个句子的两个词向量矩阵分别为static和non-static，static vector是基于word2vec中CBOW模型训练得到的词向量（不再更新），non-static词向量在模型训练过程中通过“反向传播”更新，该过程也称为Fine tune，目的是使词向量适应数据集从而提高分类效率，其输入矩阵大小为n×k。

卷积层：

输入层通过卷积操作得到若干个Feature Map，h长度的k维词向量（大小为h×k）与长度为h的卷积核进行卷积操作，h值不固定。

池化层：

文中称为Max-over-time Pooling，选择一维Feature Map中最大值（最重要的特征），最终池化层的输出为各个Feature Map的最大值，即一维向量，经过polling后大小为1×m（文中表明有m个卷积核）。

全连接 + Softmax层：

全连接层与其他卷积神经网络相同，并使用Dropout，即对全连接层上的权值参数给予L2正则化的限制，减轻过拟合程度。

**模型流程具体公式**

The *k*-dimensional word vector corresponding to the *i*-th word in the sentence 

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/2.png">

A sentence of length n is represented as

<img src="https://github.com/jm199504/Paper-Notes/blob/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/3.png">

A feature ci is generated from a window of words xi:i+h−1 by

<img src="https://github.com/jm199504/Paper-Notes/tree/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/4.png">

where b∈R is a bias term and f is a non-linear function such as the hyperbolic tangent

This filter is applied to each possible window of words in the sentence                                                      {x1:h,x2:h+1;…;xn-h+1:n}to produce a feature map

<img src="https://github.com/jm199504/Paper-Notes/tree/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/5.png">

We then apply a max-over-time pooling operation

<img src="https://github.com/jm199504/Paper-Notes/tree/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/6.png">

The penultimate layer (note that here we have m filters)

<img src="https://github.com/jm199504/Paper-Notes/tree/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/7.png">

Output unit y

<img src="https://github.com/jm199504/Paper-Notes/tree/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/8.png">

Output unit y with dropout 

<img src="https://github.com/jm199504/Paper-Notes/tree/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/9.png">

Where ◦ is the element-wise multiplication opera-tor and r∈Rm is a ‘masking’ vector of Bernoulli random variables with probability of being 1.

At test time, the learned weight vectors are scaled by p such that w^=pw and w^ is used       (without dropout) to score unseen sentences.

We additionally constrain l2-norms of the weight vectors by rescaling w to have ||w||2=s whenever ||w||2>s after a gradient descent step.

**超参数设置（基于SST-2 dev数据集和利用网格搜索）**

• Rectified linear units

• Filter windows (h) ：3, 4, 5 with 100 feature maps each

• Dropout rate (p) ：0.5

• L2 constraint (s) ：3

• Mini-batch size：50

• Training is done through stochastic gradient descent over shuffled mini-batches with the Adadelta update rule 

**四个变种模型**

• CNN-rand: 所有的word vector都是随机初始化的，可训练参数。

• CNN-static: Google的Word2Vector工具(CBOW模型)得到的结果，不再训练；

• CNN-non-static: Google的Word2Vector工具(CBOW模型)得到的结果，并在训练过程中被Fine tune；

• CNN-multichannel: CNN-static和CNN-non-static的混合版本，即两种类型的输入；

static & non-static 效果对比

• static版本：基于word2vector模型，bad最相近词为good，原因为该两个词在句法上的使用极其类似，进行简单替换不会出现语句毛病；

• non-static的版本中，bad对应的最相近词为terrible，主要因为在Fune tune的过程中，vector的值发生改变从而更加贴切数据集（情感分类数据集），因此在情感表达的角度该两个词会更加接近；

<img src="https://github.com/jm199504/Paper-Notes/tree/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/10.png">

**数据集**

• MR: Movie reviews with one sentence per review. Classification involves detecting positive/negative reviews.

• SST-1: Stanford Sentiment Treebank—an extension of MR but with train/dev/test splits provided and fine-grained labels (very positive, positive, neutral, negative, very negative), re-labeled by Socher et al.

• SST-2: Same as SST-1 but with neutral reviews removed and binary labels.

• Subj: Subjectivity dataset where the task is to classify a sentence as being subjective or objective.

• TREC: TREC question dataset—task involves classifying a question into 6 question types (whether the question is about person, location, numeric information, etc.) .


**基于TensorFlow实现的CNN文本分类代码**

http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

**模型流程**

<img src="https://github.com/jm199504/Paper-Notes/tree/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/11.png">

**CNN-multichannel model模型**

<img src="https://github.com/jm199504/Paper-Notes/tree/master/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/images/12.png">
