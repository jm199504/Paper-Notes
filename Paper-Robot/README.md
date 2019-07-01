PaperRobot Incremental Draft Generation of Scientific Ideas——阅读笔记

**论文目标：**

(1)深度理解特定领域的大量论文和构建全面的知识图谱KGs

(2)基于KGs结合图谱注意力机制和上下文注意力机制产生新的论文思路

(3)通过输入标题和相关试题生成摘要；通过摘要生成结论和未来工作；最后通过未来工作生成下一篇文章的标题

 

**现存问题/论文动机：**

(1)生物领域每年超过数十万论文被发表，人工阅读无法完成如此巨大的数量

(2)通过知识图谱可以产生新的论文思路（论文机器人无法创建新的节点，但可以利用知识图谱挖掘新的边信息）。

 

**方法：**

(1)框架总流程

图1

(2)背景知识提取

从大量生物论文提取实体entity和关系relation构造背景知识图谱 background knowledge graphs （KGs），基于Wei at提出的 an entity mention extraction and linking system方法提取3种实体类别（Disease, Chemical and Gene）和133种边关系。

(3)链路预测

在已有文献构建初始的KGs后，通过链路预测强化KGs，即完善实体信息表示，并且通过实体信息的表示来判断两实体间在语义上是否相似，例如图中钙和锌在上下文信息和图结构上相似，这样我们可以预测钙的两个新邻居（CD14 molecule和neuropilin 2）。

1.初始化KGs：e(h,i) 为头实体，e(t,i)为尾实体，r(i)为关系，以及N(e(i))=[n(i,1),n(i,2),…]为one-hop连接邻居[如上图钙与CD14]，e(i)和上下文描述s(i)有所关联，s(i)是随机选择出现了e(i)的语句，分别随机设定e(i),r(i)的向量初始值。

2.图结构编码：为了捕获邻居节点特征对e(i)的重要性，基于Velickovi提出的自注意力self-attention和计算N(e(i))的权重分布：(即one-hop邻居对e(i)节点的影响重要程度)

图3

其中⊕表示矩阵连接操作concatenation operation，利用c(i)'和N(e(i))计算实体结构表示：

图4

其中σ为sigmoid函数。

为了捕捉e(i)和邻居的多样关系，对每个实体采用了多头注意力机制multi-head attention，即多种线性转换矩阵，获得实体最终结构表示：

图5

其中ε(m,i)表示第m个头计算的结构表示。

3.上下文编码：每一个实体均有上下文语句[w1,w2,…,wl]，为了囊括文本信息，采用Graves和Schmidhuber提出的双向LSTM模型获得编码隐藏单元状态H(s)=[h1,h2,..hl]，其中h1表示w1的隐藏状态，计算每个词w的双线性注意力bilinear：

图6

其中W(s)为biliear term，计算实体最终上下文表示：

图7

4.门控组合：基于图的表示e^和局部上下文表示e~，设计了一个门函数权衡这2个实体表示：

备注：门控可理解为对信息传输的限流，丢弃或者保留哪一部分信息，其越接近0表示几乎全部丢弃；门函数如sigmoid，tanh等

图8

其中ge是各元素范围[0,1]间的实体依赖门函数，ge~是对于实体e可学习参数，σ是sigmoid， ·○是元素对应相乘。

5.训练和预测：优化实体和关系的表示，基于Bordes提出的TransE假设两个实体之间的关系被解释为对实体的翻译表示，边际损失函数训练模型：

图9

其中F函数-两实体间距离得分:

图10

其中(e(i,h),r(i),e(i,t))表示正元组，另一为负元组，γ为边际margin，通过随机选择不同实体替换正元组的头部或尾部实体来生成负元组。

计算分数y表示(ei; ri; ej) 把握概率，获得增强版的背景图谱：

图11

☆链路预测-小结：

①初始化背景图谱KGs

②对实体和one-hop邻居节点使用自注意力机制和多头注意力机制实现对实体的图结构表示

③对实体和上下文语句使用双向LSTM和bilinear注意力实现对实体的上下文表示

④使用门控组合权衡实体的2种表示

⑤通过训练和预测生成增强版的背景图谱KGs

 

(4)写新things

1.参考编码器：将参考标题每一个词随机嵌入式表达为向量τ = [w1;…; wl]，使用Cho提出的双向GRU编码器生成隐藏状态H=[h1,…,hl]

2.初始化解码器：从最后一个隐藏状态编码和相关实体E = [e1;…; ev]，使用多跳注意力机制生成初始化隐藏状态解码

图12

其中hl为作为查询向量q0，k表示第k个跳跃。

3.记忆网络：使用记忆网络提取每个实体的细粒度权重和记忆上下文向量：

图13

其中cij^：

图14

4.参考注意力：使用与Seq2seq和Pointer Network相似的参考注意力，捕获每个词解码输出的贡献，在每一个时间步i中，解码器会接收前一个词的嵌入和生成解码器状态h(i)，每个参考token的注意力权重和参考上下文向量：

图15

其中c(ij)是参考覆盖向量，see提出的对所有前层解码器注意力分布求和以减少重复，φi是参考文本向量：

图16

5.生成器：对于特定的单词w，它可能在引用标题或多个相关实体中出现多次。因此，在每个解码步骤i中，对于每个单词w，我们从参考注意分布和记忆注意分布中汇总其注意权重

汇总参考注意力分布：

图17

汇总记忆注意力分布：

图18

基于语言模型的词概率：

图19

第i步token z的最终概率：

图20

See提出的结合参考注意力和记忆分布的覆盖损失函数：

图21

6.重复删除：我们使用覆盖损失(基于覆盖机制)避免实体在引用输入和相关实体重复，并使用波束搜索生成输出值

 

**☆写新things-小结：**

①使用双向GRU编码器利用参考标题生成隐藏状态编码(encoder hidden states)

②使用多跳跃注意力机制对最后一个隐藏状态编码hl和相关实体E生成初始化的隐藏状态解码(decoder hidden states)

③构建带有coverage机制的记忆网络提取记忆注意力分布(Memory attention distributions)，最终生成记忆上下文向量(memory based context vector)

④使用Seq2seq和Pointer Network相似的参考注意力获取参考注意力分布(reference attention distributions)，最终生成参考上下文向量(reference context vector)

⑤分别汇总参考注意力分布和记忆注意力分布中的注意权重记作为Pt和Pe，基于记忆上下文向量和参考上下文向量生成Pgen

⑥使用双门结合Pt,Pe,Pgen生成token z的最终概率

⑦结合带有coverage loss的损失函数和集束搜索处理实体重复问题

 

 

**涉及方法/模型：**

1.基于Wei at提出的 An entity mention extraction and linking system

2.基于Velickovi提出的自注意力self-attention

3.多头注意力机制multi-head attention

图22

4.Graves和Schmidhuber提出的双向LSTM模型 bi-directional long short-term memory (LSTM) 

5.双线性注意力bilinear attention

6.门控组合Gated Combination

门控组合是指控制多源信息的传递，另外门函数如sigmoid，tanh等

7.基于Bordes提出的TransE

8.Cho提出的双向GRU编码器bi-directional Gated Recurrent Unit (GRU) encoder

9.多跳注意力机制multihop attention mechanism

10.与Seq2seq和Pointer Network相似的参考注意力

11.覆盖机制 coverage mechanism

12.集束搜索 beam search 

13.隐藏状态 hidden state

隐藏状态可以理解为神经网络的记忆，包含了先前的数据信息，例如LSTM中的ht

图23

