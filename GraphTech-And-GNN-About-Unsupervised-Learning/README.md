## 图技术及图神经网络相关的无监督学习算法

1.社区发现算法

传统算法：CPM；LPA；Fast Unfolding；HANP；LFM；SLPA；BMLPA；COPRA

思想：以网络结构信息作为聚类标准，以模块度作为优化目标

局限性：仅利用了网络结构信息，没有涉及节点自身特征

基础概念：发现网络中的社区结构

工程应用：社会关系网络、资金链网络、城市交通网络

社区（community）定义：子图（包含顶点和边），同社区内节点连接紧密，社区间连接稀疏

社区划分类型：重叠社区和非重叠社区，其中重叠社区表示节点可由多个社区标注，即某节点含多个标签

2.DeepWalk

思想：基于NLP对节点随机游走构建节点的网络结构信息向量表示

局限性：仅利用了网络结构信息，没有涉及节点自身特征

优势：支持在线学习可扩展，并行计算

3.谱聚类

思想：基于图论的聚类方法

局限性：传统的谱聚类仅利用了网络结构信息，没有涉及节点自身特征

改进：使用节点间的特征相似度代替原有连通设为1的邻接矩阵可以结合节点特征和网络结构信息

4.LINE

思想：利用一阶相似度和二阶相似度对网络节点表达

局限性：仅利用了网络结构信息，没有涉及节点自身特征

优势：挖掘更多的网络节点相似性度量标准

5.LargeVis

思想：基于LINE的高维数据可视化算法（对LINE算法的改进）

6.图自编码GAE和变分图自编码VGAE

思想：基于图神经网络对网络节点表达

优势：学习了网络节点的自身特征和网络结构信息

代码：https://github.com/tkipf/gae

7.反正则化图自编码器ARGA和反正则化变分自编码器ARVGA

思想：基于图神经网络对网络结构重构

优势：在重构过程中不断优化网络节点的表达

代码：https://github.com/Ruiqi-Hu/ARGA

社区发现简单归纳：

| No   | Algorithm                                             | 时间 | 类别     |
| ---- | ----------------------------------------------------- | ---- | -------- |
| 1    | CPM（Cluster   Percolation method）                   | 2005 | 重叠社区 |
| 2    | LAP（Label   propagation algorithm）                  | 2007 | 非重叠   |
| 3    | Fast Unfolding                                        | 2008 | 非重叠   |
| 4    | HANP（Hop   Attenuation & Node Preference）           | 2009 | 非重叠   |
| 5    | LFM（Latent   factor model）                          | 2009 | 重叠社区 |
| 6    | SPLA（Speak-listener   Label propagation algorithm）  | 2011 | 重叠社区 |
| 7    | BMLPA（Balanced   Multi-label propagation algorithm） | 2013 | 非重叠   |
| 8    | COPRA                                                 | -    | 重叠社区 |
| 9    | GN（Girvan-Newman）                                   | -    | 非重叠   |

**图技术相关论文调研**

Graph Attention Networks

主要内容：提出基于近邻节点注意力机制的网络模型graph attention networks (GATs)来解决图卷积方法和基于谱的神经网络（spectral-based graph neural networks）的缺陷，可用于处理复杂、不规则的计算图。

Graph Partition Neural Networks for Semi-Supervised Classification

主要内容：提出图分割神经网络（Graph Partition Neural Network，GPNN）可适用于快速处理大型图（extremely large graphs），其子图的节点间信息局部传播和子图间信息全局传播交替进行。

代码：https://github.com/Microsoft/graph-partition-neural-networksamples

Stochastic Training of Graph Convolutional Networks with Variance Reduction

主要内容：提出基于控制变量算法（control variate based algorithms）的图卷积网络（GCN）可以有效减少感受野大小，收敛速度更快。传统GCN节点表示来源于其邻接点的迭代，使得感受野随着层数呈指数爆炸增长，前期尝试下采样减少其感受野大小却难以保证收敛，并且它们每个节点的感受野大小仍然数百。控制变量算法可对任何少的邻接点数量进行采样。

Adaptive Graph Convolutional Neural Networks

主要内容：提出自适应图卷积神经网络 （Adaptive Graph Convolutional Neural Networks，AGCN）可接收任意图结构和规模的图作为输入，图在训练过程中可以学到相应的自适应图，其收敛速度和预测精准度均有提升。

Adversarial Attacks on Neural Networks for Graph Data

主要内容：提出了针对图深度学习模型的对抗攻击方法，是首个在属性图（attributed graphs）上的对抗攻击研究，为了应对潜在的离散域还提出了一种利用增量计算的高效算法 Nettack。

Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition

主要内容：提出了一种时空图卷积网络（SpatialTemporal Graph Convolutional Networks (ST-GCN)）捕获数据中的时空模式，使模型更强的表达能力和泛化能力。

Learning Human-Object Interactions by Graph Parsing Neural Networks

主要内容：提出图解析神经网络（Graph Parsing Neural Network，GPNN），用于检测和识别图像和视频中人-物交互（human-object interactions (HOI)）的任务。

GPNN推断出包含的解析图：邻接矩阵的HOI图结构+节点标签组成。

GPNN会迭代计算邻接矩阵和节点标签。

Graph Convolution over Pruned Dependency Trees Improves Relation Extraction

主要内容：提出一种用于关系提取的图搜索神经网络（Graph Search Neural Network (GSNN)），通过知识图获取结构化的先验知识作为额外信息来提升图像分类效果。

**应用场景：**

molecular data

point could

social networks

**社区发现数据集下载：**

**基于链接分析的数据集**

（1）Zachary karate club

Zachary 网络是通过对一个美国大学空手道俱乐部进行观测而构建出的一个社会网络.网络包含 34 个节点和 78 条边,其中个体表示俱乐部中的成员,而边表示成员之间存在的友谊关系.空手道俱乐部网络已经成为复杂网络社区结构探测中的一个经典问题[1]。

链接：<http://www-personal.umich.edu/~mejn/netdata/karate.zip>

（2）College Football 

College Football 网络. Newman 根据美国大学生足球联赛而创建的一个复杂的社会网络.该网络包含 115个节点和 616 条边,其中网络中的结点代表足球队,两个结点之间的边表示两只球队之间进行过一场比赛.参赛的115支大学生代表队被分为12个联盟。比赛的流程是联盟内部的球队先进行小组赛,然后再是联盟之间球队的比赛。这表明联盟内部的球队之间进行的比赛次数多于联盟之间的球队之间进行的比赛的次数.联盟即可表示为该网络的真实社区结构[2]。

链接：<http://www-personal.umich.edu/~mejn/netdata/football.zip>

（3）Dolphin

Dolphin 数据集是 D.Lusseau 等人使用长达 7 年的时间观察新西兰 Doubtful Sound海峡 62 只海豚群体的交流情况而得到的海豚社会关系网络。这个网络具有 62 个节点，159 条边。节点表示海豚，而边表示海豚间的频繁接触[3]。

链接：<http://www-personal.umich.edu/~mejn/netdata/dolphins.zip>

**基于链接与离散型属性的数据集**

（1）Political blogs

该数据集由Lada Adamic于2005年编译完成， 表示博客的政治倾向。 包含1490个结点和19090条边。数据集中的每个结点都有一个属性描述（用0或者1表示），表示民主或者保守[4] 。

链接：<http://www-personal.umich.edu/~mejn/netdata/polblogs.zip>

**基于链接与文本型属性的数据集**

（1）Enron Email Dataset

该数据集是由CALO项目(一个学习和组织的认知助手)收集和准备的。它包含了大约150名用户的数据，其中大部分是安然的高级管理人员，这些数据被组织到[7]文件夹中。

链接：<http://www.cs.cmu.edu/~enron/enron_mail_20150507.tgz>

（2）Cora

Cora数据集由2708份科学出版物组成，共分为7类。引文网络由5429个链接组成。数据集中的每个发布都由一个0/1值的单词向量描述，该向量表示字典中对应单词的缺失/存在。这本词典由1433个独特的单词组成。数据集中的README文件提供了更多的细节[8]。

链接：<http://www.cs.umd.edu/~sen/lbc-proj/data/cora.tgz>

（3）WebKB

WebKB数据集包含877种科学出版物，分为5类。引文网络由1608个链接组成。数据集中的每个发布都由一个0/1值的单词向量描述，该向量表示字典中对应单词的缺失/存在。这部词典由1703个独特的单词组成。数据集中的README文件提供了更多的细节[9]。

链接：<http://www.cs.umd.edu/~sen/lbc-proj/data/WebKB.tgz>

（4）Terrorists & Terrorist Attacks

第2个数据集包括1293起恐怖袭击，每起袭击都有6个标签，每个标签表示袭击的类型。每个攻击都由一个0/1值的属性向量描述，其条目表示某个特性的不存在。共有106个不同的特征。数据集中的文件可用于创建两个不同的图。数据集中的README文件提供了更多细节。

链接：<http://www.cs.umd.edu/~sen/lbc-proj/data/TerroristRel.tgz>

链接：<http://www.cs.umd.edu/~sen/lbc-proj/data/TerrorAttack.tgz>

**斯坦福大型网络数据集收集**

链接：<http://snap.stanford.edu/data/>

**社交网络信息分析**

KDD Cup Dataset：<http://www.cs.cornell.edu/projects/kddcup/datasets.html>

Stack Overflow Data：<http://blog.stackoverflow.com/2009/06/stack-overflow-creative-commons-data-dump/>

Youtube dataset：<http://netsg.cs.sfu.ca/youtubedata/>

Amazon Data：<http://snap.stanford.edu/data/amazon-meta.html>

**其他公开数据集：**

1）The Cora

2）Citeseer

3）Pubmed citation network datasets

4）A protein-protein interaction dataset

1.医学影像数据

医学图书馆向13,000名患者注释提供了53,000张医学图像的MedPix®数据库。需要注册。

信息：https : //medpix.nlm.nih.gov/home

ABIDE：自闭症脑成像数据交换：对自闭症内在大脑结构的大规模评估。

539名患有ASD和573名典型对照的个体的功能MRI图像。这些1112数据集由结构和静息状态功能MRI数据以及广泛的表型信息组成。需要注册。

论文：http: //www.ncbi.nlm.nih.gov/pubmed/23774715

信息：http : //fcon_1000.projects.nitrc.org/indi/abide/

预处理版本：http：// preprocessed-connectomes-project。组织/遵守/

阿尔茨海默病神经成像倡议（ADNI）

MRI数据库阿尔茨海默病患者和健康对照。还有临床，基因组和生物制造商的数据。需要注册。

论文：http : //www.neurology.org/content/74/3/201.short

访问：http : //adni.loni.usc.edu/data-samples/access-data/

用于血管提取的数字视网膜图像（DRIVE）

DRIVE数据库用于比较研究视网膜图像中血管的分割。它由40张照片组成，其中7张显示出轻度早期糖尿病视网膜病变迹象。

论文：http : //www.isi.uu.nl/Research/Publications/publicationview/id=855.html

访问：http : //www.isi.uu.nl/Research/Databases/DRIVE/download.php

AMRG心脏地图集 AMRG心脏MRI地图集是奥克兰MRI研究组西门子Avanto扫描仪采集的正常患者心脏的完整标记的MRI图像集。该地图集旨在为大学和学校的学生，MR技术人员，临床医生提供...

先天性心脏病（CHD）地图集 先天性心脏病（CHD）地图集代表成年人和患有各种先天性心脏病的儿童的MRI数据集，生理临床数据和计算机模型。这些数据来自几个临床中心，包括Rady ...

通过磁共振成像评估确定除颤器降低风险，是一项前瞻性，多中心，随机临床试验，用于冠心病和轻中度左心室功能不全患者。主要目标......

MESA 多种族动脉粥样硬化研究是一项在美国的六个中心进行的大规模心血管人群研究（> 6,500名参与者）。它的目的是调查亚临床到临床心血管疾病的表现之前......

OASIS 开放获取系列影像研究（OASIS）是一项旨在使科学界免费提供大脑核磁共振数据集的项目。两个数据集可用：横截面和纵向集。

年轻，中老年，非痴呆和痴呆老年人的横断面MRI数据：该组由416名年龄在18岁至96岁的受试者组成的横截面数据库组成。对于每位受试者，单独获得3或4个单独的T1加权MRI扫描包括扫描会话。受试者都是右撇子，包括男性和女性。100名60岁以上的受试者已经临床诊断为轻度至中度阿尔茨海默病（AD）。此外，还包括一个可靠性数据集，其中包含20个未删除的主题，在其初次会议后90天内的后续访问中成像。

非痴呆和痴呆老年人的纵向磁共振成像数据：该集合包括150名年龄在60至96岁的受试者的纵向集合。每位受试者在两次或多次访视中进行扫描，间隔至少一年，总共进行373次成像。对于每个受试者，包括在单次扫描期间获得的3或4次单独的T1加权MRI扫描。受试者都是右撇子，包括男性和女性。在整个研究中，72名受试者被描述为未被证实。包括的受试者中有64人在初次就诊时表现为痴呆症，并在随后的扫描中仍然如此，其中包括51名轻度至中度阿尔茨海默病患者。另外14名受试者在初次就诊时表现为未衰退，随后在随后的访视中表现为痴呆症。
访问：http : //www.oasis-brains.org/

SCMR共识数据 SCMR共识数据集是从不同的MR机（4个GE，5个西门子，6个Philips）获得的混合病理学（5个健康，6个心肌梗塞，2个心力衰竭和2个肥大）的15个心脏MRI研究）。主要目标......

Sunnybrook心脏数据 Sunnybrook心脏数据（SCD）也被称为2009年心脏MR左心室分割挑战数据，由45个病人和病理混合的电影-MRI图像组成：健康，肥大，伴有梗塞和心脏的心力衰竭。 ..

访问：http : //www.cardiacatlas.org/studies/

肺图像数据库联盟（LIDC）

初步的临床研究表明，螺旋CT扫描肺部可以提高高危人群的肺癌早期发现率。图像处理算法有可能有助于螺旋CT研究中的病变检测，并评估连续CT研究中病变大小的稳定性或变化。这种计算机辅助算法的使用可以显着提高螺旋CT肺部筛查的灵敏度和特异性，并且通过减少解释所需的医师时间来降低成本。

肺成像数据库联盟（LIDC）倡议的目的是支持一个机构联盟制定螺旋CT肺部影像资源的共识指南，并建立螺旋CT肺部影像数据库。根据这项计划资助的研究人员为数据库的使用创建了一套指导方针和指标，并为开发数据库作为实验台和展示这些方法的指南和指标。该数据库通过互联网向研究人员和用户提供，作为研究，教学和培训资源具有广泛的用途。

具体而言，LIDC倡议的目标是提供：

用于图像处理或CAD算法的相对评估的参考数据库
一个灵活的查询系统，将为研究人员提供评估各种技术参数的机会，并取消确定该数据库中的临床信息，这对研究应用很重要。
该资源将刺激进一步的数据库开发，用于包括癌症筛查，诊断，图像引导干预和治疗在内的应用的图像处理和CAD评估。因此，NCI鼓励研究者发起的拨款申请，在他们的研究中利用数据库。NCI还鼓励研究者发起的赠款申请，这些申请提供了可以改进或补充LIDC使命的工具或方法。

访问：http : //imaging.cancer.gov/programsandresources/informationsystems/lidc

TCIA集合

跨各种癌症类型（例如癌，肺癌，骨髓瘤）和各种成像模式的癌症成像数据集。“癌症成像档案”（TCIA）中的图像数据被组织成特定目标的主题集合。受试者通常具有癌症类型和/或解剖部位（肺，脑等）。下表中的每个链接都包含有关集合的科学价值的信息，关于如何获取任何可用的支持非图像数据的信息以及查看或下载成像数据的链接。为了支持科学研究的可重复性，TCIA支持数字对象标识符（DOI），允许用户共享研究手稿中引用的TCIA数据的子集。

访问：http : //www.cancerimagingarchive.net/

白俄罗斯结核病门户

结核病（TB）是白俄罗斯公共卫生的一个主要问题。最近的情况与MDR / XDR结核病和HIV / TB需要长期治疗的出现和发展相关。许多和最严重的病例通常在全国各地传播到不同的结核病药房。通过使用包含患者放射影像，实验室工作和临床数据的共同数据库，领先白俄罗斯结核病专家关注这些患者的能力将大大提高。这也将显着改善对治疗方案的依从性，并且更好地记录治疗结果。纳入门诊患者入选临床病例的标准 - 入住肺结核和肺结核的RDSC耐多药结核病部门，诊断或怀疑患有耐多药结核病，

访问：http : //tuberculosis.by/

DDSM：用于筛选乳腺摄影的数字数据库

乳腺摄影数字化数据库（DDSM）是乳腺摄影图像分析研究社区使用的资源。该项目的主要支持来自美国陆军医学研究和装备司令部的乳腺癌研究计划。DDSM项目是由马萨诸塞州综合医院（D. Kopans，R. Moore），南佛罗里达大学（K. Bowyer）和桑迪亚国家实验室（P. Kegelmeyer）共同参与的合作项目。华盛顿大学医学院的其他病例由放射学和内科医学助理教授Peter E. Shile博士提供。其他合作机构包括威克森林大学医学院（医学工程和放射学系），圣心医院和ISMD，Incorporated。数据库的主要目的是促进计算机算法开发方面的良好研究，以帮助筛选。数据库的次要目的可能包括开发算法以帮助诊断和开发教学或培训辅助工具。该数据库包含约2,500项研究。每项研究包括每个乳房的两幅图像，以及一些相关的患者信息（研究时间，ACR乳房密度评分，异常微妙评级，异常ACR关键字描述）和图像信息（扫描仪，空间分辨率... ）。包含可疑区域的图像具有关于可疑区域的位置和类型的像素级“地面真实”信息。

访问：http : //marathon.csee.usf.edu/Mammography/Database.html

前列腺

据报道，前列腺癌（CaP）在全球范围内是第二大最频繁诊断的男性癌症，占13.6％（Ferlay等（2010））。据统计，2008年，新诊断病例的数量估计为899,000，其中不少于258,100例死亡（Ferlay等（2010））。

磁共振成像（MRI）提供成像技术，可以诊断和定位CaP。I2CVB提供多参数MRI数据集以帮助开发计算机辅助检测和诊断（CAD）系统。访问：http : //i2cvb.github.io/

访问：http : //www.medinfo.cs.ucy.ac.cy/index.php/downloads/datasets

多发性硬化症数据库中的MRI病灶分割

紧急远程骨科X射线数字图书馆

IMT分割

针EMG MUAP时域特征

DICOM图像样本集 这些数据集专门用于研究和教学。您无权重新发布或出售它们，或将其用于商业目的。

所有这些DICOM文件都使用JPEG2000传输语法进行压缩。

访问：http : //www.osirix-viewer.com/resources/dicom-image-library/

SCR数据库：胸部X光片的分割

胸部X光片中解剖结构的自动分割对于这些图像中的计算机辅助诊断非常重要。SCR数据库的建立是为了便于比较研究肺野，心脏和锁骨在标准的后胸前X线片上的分割。

本着合作科学进步的精神，我们可以自由共享SCR数据库，并致力于在这些分割任务上维护各种算法结果的公共存储库。在这些页面上，可以在下载数据库和上载结果时找到说明，并且可以检查各种方法的基准结果。

访问：http : //www.isi.uu.nl/Research/Databases/SCR/

医学影像数据库和图书馆

访问：http : //www.omnimedicalsearch.com/image_databases.html

一般类别

e-Anatomy.org - 交互式解剖学图谱 - 电子解剖学是解剖学在线学习网站。为了覆盖人体的整个断面解剖结构，选择了来自正常CT和MR检查的超过1500个切片。图像使用Terminologia Anatomica标记。用户友好的界面允许通过结合交互式文本信息，3D模型和解剖图绘制的多切片图像系列进行摄影。

医学图片和定义 - 欢迎访问互联网上最大的医学图片和定义数据库。有许多网站提供医疗信息，但很少提供医疗照片。就我们所知，我们是唯一一家提供医学图片数据库的关于每个术语的基本信息的图片。编者按：好的网站可免费访问，无需注册1200多种健康和医疗相关图像，并带有定义。

核医学艺术 - 医学插图，医学艺术。包括3D动画。“Nucleus Medical Art，Inc.是美国和海外的出版，法律，医疗，娱乐，制药，医疗设备，学术界和其他市场的医疗插图，医疗动画和交互式多媒体的领先创造者和分销商。注意：伟大的网站。

互联网上的医学图像数据库（UTHSCSA Library） - 指向具有主题特定医疗相关图像的网站的链接目录。

手术视频 - 国家医学图书馆MedlinePlus收集100和100s不同外科手术的链接。您必须在电脑上安装RealPlayer媒体播放器才能观看这些免费的视频。

带插图的ADAM医学百科全书。也许今天互联网上最好的插图医学着作之一，ADAM医学百科全书收录了4000多篇有关疾病，测试，症状，受伤和手术的文章。它还包含一个广泛的医学照片和插图库，用于备份这4,000篇文章。这些插图和文章免费向公众开放。

哈丁医学博士 - 医学和疾病图片，是一个由爱荷华大学提供的相当一段时间的免费和已建立的资源。主页处于目录风格，用户将不得不深入查找他们正在查找的图像，其中许多图像不在现场。尽管如此，哈丁医学博士是一个很好的门户，可以查看数千种详细的医疗照片和插图。

健康教育资产图书馆（HEAL） - 网络健康基金会媒体库总部位于瑞士的（HON）是一个国际机构，旨在鼓励在线健康信息的道德提供。“HONmedia（图像库）是一个超过6,800个医学图像和视频的独特库，涉及1,700个主题和主题。这个无与伦比的数据库由HON手动创建，新图像链接不断从全球范围添加HON鼓励用户通过提交图片链接制作自己的图片链接。“ 图书馆包括解剖图像，疾病和条件以及程序的视觉影响。

公共卫生图像库（PHIL）由疾病控制和预防中心（CDC）的工作组创建，PHIL为CDC的图片提供了一个有组织的通用电子网关。我们欢迎公共卫生专业人员，媒体，实验室科学家，教育工作者，学生和全球公众使用这些材料作为参考，教学，演示和公共卫生信息。内容被组织成人物，地点和科学等级分类，并以单幅图像，图像集和多媒体文件形式呈现。

医学史图片 - 该系统提供了美国国家医学图书馆（NLM）医学史分部（HMD）的印刷品和图片集中近6万幅图片的访问权限。该系列包括各种媒体的肖像，机构图片，漫画，流派场景和平面艺术，展示了医学的社会和历史方面。

Pozemedicale.org - 以西班牙语，意大利语，葡萄牙语和意大利语收集医学图像。

旧医学图片：从19世纪末和20世纪初，数百个迷人而有趣的旧，但高品质的照片和图像。

学科专业图像库和集合

亨利·格雷的人体解剖 - 格雷的人体解剖学Bartleby.com版以经典的1918年出版物中的1,247幅鲜艳的雕刻 - 许多颜色为特征。

Crookston系列 - 由John H. Crookston博士拍摄的医学幻灯片集合，已经数字化，可供公众和医生使用。

DAVE项目 - 涵盖广谱内窥镜成像的胃肠内窥镜视频剪辑的可搜索库。

Dermnet - 可收集超过8000种高品质皮肤科图像。

交互式皮肤科Atlas - 常见和罕见皮肤问题的图像参考资源。

多维人类胚胎是由国家儿童健康与人类发育研究所（NICHD）资助的一项合作，旨在通过互联网制作并提供基于磁共振成像的人类胚胎的三维图像参考。

GastroLab内窥镜档案于1996年发起，目标是保持内窥镜图库免费供所有感兴趣的医护人员使用。

MedPix是放射学和医学图片数据库资源工具。主页界面很混乱，整个网站设计不友好，并且在20世纪90年代中期给它留下了印象。但是，如果你有时间（耐心），它可能被证明是一些重要的资源。

OBGYN.net图像库 - 本网站致力于提供对女性健康感兴趣的图像。除了为您提供访问OBGYN.net图像外，我们还指出了互联网上其他女性健康相关的图像。由于材料的图形性质，有些人可能不喜欢看这些图像。它们仅用于教育目的。

威盛集团公共数据库

记录图像数据库对于定量图像分析工具的开发至关重要，特别是对于计算机辅助诊断（CAD）的任务。与I-ELCAP小组合作，我们建立了两个公共图像数据库，其中包含DICOM格式的肺部CT图像以及放射科医师的异常记录。请访问下面的链接了解更多详情：

访问：http：//www.via.cornell.edu/databases/

CVonline：图像数据库 访问：http : //homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm

USC-SIPI图像数据库 USC-SIPI图像数据库是数字化图像的集合。它主要用于支持图像处理，图像分析和机器视觉方面的研究。USC-SIPI图像数据库的第一版于1977年发布，并且自那时以来增加了许多新图像。

数据库根据图片的基本特征分为多个卷。每个卷中的图像具有各种尺寸，例如256x256像素，512x512像素或1024x1024像素。所有图像的黑白图像均为8位/像素，彩色图像为24位/像素。目前提供以下卷：

Textures         Brodatz textures, texture mosaics, etc.Aerials         High altitude aerial imagesMiscellaneous         Lena, the mandrill, and other favoritesSequences         Moving head, fly-overs, moving vehicles
访问：http : //sipi.usc.edu/database/

2.挑战/比赛数据
放射学中的视觉概念提取挑战 手动注释来自几种不同成像模式（例如CT和MR）的几种解剖结构（例如肾，肺，膀胱等）的放射学数据。他们还提供了一个云计算实例，任何人都可以使用它来根据基准开发和评估模型。

访问：http：//www.visceral.eu/

生物医学图像分析中的重大挑战

通过标准化评估标准，为了便于在新解决方案和现有解决方案之间进行更好的比较，收集生物医学成像挑战。您也可以创建自己的挑战。截至撰写本文时，有92个挑战提供可下载的数据集。

访问：http : //www.grand-challenge.org/

梦想的挑战

梦想的挑战提出了关于系统生物学和转化医学的基本问题。我们的挑战由来自各种组织的研究人员社区设计和运行，邀请参与者提出解决方案 - 促进协作并在此过程中建立社区。Sage Bionetworks提供专业技术和制度支持，以及通过Synapse平台应对挑战的基础设施。我们共同拥有一个愿景，允许个人和团体公开合作，使“人群中的智慧”对科学和人类健康产生最大的影响。

数字乳腺摄影梦想挑战。
ICGC-TCGA DREAM体细胞突变称为RNA挑战（SMC-RNA）
梦想的挑战
这些是在增加时面临的积极挑战，还有更多过去的挑战和即将到来的挑战！
访问：http : //dreamchallenges.org/

(参考来源：https://blog.csdn.net/weixin_41923961/article/details/80547291)
**调研细节**

**1.图技术中的无监督学习算法：**

（1）社区发现（Community Detection）：

特点：以网络结构信息作为聚类标准，以模块度作为优化目标，如经典算法：LPA、CPM、HANP等

内容：9种经典社区发现算法

1.CPM派系过滤算法

k派系表示所有节点两两相连(派系=完全子图)

k派系相邻：2个不同k派系共享k-1个节点

k派系连通：1个k派系可以通过若干个相邻k派系到达另1个k派系

步骤：

- 1.寻找大小为k的完全子图
- 2.每个完全子图定义为一个节点，建立重叠矩阵[对称]（对角线元素均为k，派系间值为共享节点数）
- 3.将重叠矩阵中非对角线元素小于k-1值置为0
- 4.划分社区

2.LPA

步骤：

- 1.所有节点初始化唯一标签

- 2.基于邻节点最多标签更新节点（最多个数标签不唯一时随机）

- 备注：涉及同步更新和异步更新：

- 同步更新：节点z在第t次迭代依据于邻居节点第t-1次label

- 异步更新：节点z在第t次迭代依据于邻居节点第t次label和邻居节点第t-1次label（部分更新）

3.Fast Unfolding

基于模块化modularity       optimization启发式方法

步骤：

- 1.每个节点不断划分到其他节点的社区使模块度最大化
- 2.将已划分社区再重复以上思想进行聚类
- 备注：类似于层次聚类

4.HANP

思想：传播能力随着传播距离增大而减弱

5.LFM

思想：定义节点连接紧密程度函数，并使社区内该函数值越大

6.SLPA

思想：对LPA的优化算法，保留LPA每个节点的更新记录，并选择出现次数最大的类

7.BMLPA

思想：定义归属系数，传播计算节点属于周围邻节点的概率

8.COPRA

思想：类似于BMLPA，且每个节点归属于类的概率之和为1

9.GN

思想：不断删除网络中相对于所有源节点最大的边介数的边
边介数betweenness：网络中任意两个节点通过该边的最短路径的数量
GN算法的步骤如下： 

- 1.计算每一条边的边介数； 

- 2.删除边界数最大的边； 

- 3.重新计算网络中剩下的边的边阶数；

- 4.重复(3)和(4)步骤，直到网络中的任一顶点作为一个社区为止。

 
（2）DeepWalk：

论文：DeepWalk: Online Learning of Social Representations（2014）

特点：仅涉及网络结构信息，支持在线学习可扩展，并行计算

目标：获得节点出现的概率分布和学习节点表示

数据集：BlogCatalog/Flickr/YouTube

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/1.png">

相关知识：

1）所有的语言模型中需要输入语料库corpus和词vocabulary，拓展到DeepWalk中，而random walk是corpus，graph中的节点为vocabulary

2）采用distribute representation词向量编码，低维实数向量，通常为50维/100维，其最大贡献是相关或者相似词在距离上更近，其距离可以使用余弦相似度或欧氏距离，其通常被称呼为word representation / word embedding.

3）对应one-hot具有维度过大的问题，词的维度即语料库大小

4）Skip-Gram Model是根据某个词分别计算它前后出现某几个词的各个概率。[与其对应的是CBOW模型∈word2vec]

5）Hierarchical Softmax用Huffman编码构造二叉树，其实借助了分类问题中，使用一连串二分类近似多分类的思想。

6）作者不采用CBOW的原因认为基于前后N个词去表示单词的计算量较大

CBOW & Skip-Gram模型 ，而采用SkipGram

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/2.png">

内容：基于NLP中word2vec思想运用到网络的节点表示中，对图中节点进行一串随机游走应用到网络的表示(随机游走序列)，节点类似于词向量，节点间游走类似于句子，从截断的随机游走序列中得到网络的局部信息，再通过局部信息来学习节点的潜在表示。

模型流程：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/3.png">

算法：

1-使用Random walk随机游走生成节点序列

2-遍历所有的节点（SkipGram / Hierarchical Softmax（to speed the training time）/ Optimization(SGD and BP)）

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/4.png">

其中t表示长度；γ为迭代次数；

基于以下公式更新SkipGram算法：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/5.png">

SkipGram:其目的最大化出现在上下文的所有单词的概率以更新向量表示

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/6.png">

其中Φ(vj)是对vj的向量表示

3-为了加速模型训练效率，使用Hierarchical Softmax，在计算Pr(uk|Φ(vj))需要遍历语料库统计uk和vj同时出现的概率，而语料库极大，使得时间复杂度极高，因此使用二叉树进行优化（Huffman树），每个节点在随机游走路径中出现次数作为权重，其根节点是输入节点，非叶子节点为隐层权重，所有的隐层权重及叶子节就是我们需要更新的参数。

其输入vj，Pr(uk|Φ(vj))就是最大化从vj到uk所有路径相乘的概率，二叉树的每一个分支均为二分类问题。

代码：https://github.com/phanein/deepwalk

（3）谱聚类（Spectral Clustering）：

论文：Parallel Spectral Clustering in Distributed Systems（IEEE 2010）

特点：基于图论的聚类方法，仅涉及网络结构信息

内容：利用邻接矩阵A和度矩阵D生成拉普拉斯矩阵L并归一化，计算矩阵L的特征向量和特征值

本质：通过拉普拉斯矩阵变换得到其特征向量组成的新矩阵的 K-Means 聚类，而其中的拉普拉斯矩阵变换可以被简单地看作是降维的过程。而谱聚类中的「谱」其实就是矩阵中全部特征向量的总称。

举例说明：

第一步：构建邻接矩阵A（相连为1，不相连为0）

第二步：构建度矩阵D

第三步构建拉普拉斯矩阵L，L=D-A

备注：拉普拉斯矩阵的构造是为了将数据之间的关系反映到矩阵中

第四步归一化矩阵L即：

第五步计算矩阵L的特征值和特征向量(谱分解)

备注：计算特征值以及特征向量从而达到将维度从N维降到k维

第六步即可采用如K-means聚类

总结：我们要将N维的矩阵压缩到k维矩阵，那么就少不了特征值，取前k个特征值进而计算出k个N维向量P(1),P(2),...,P(k).这k个向量组成的矩阵N行k列的矩阵（经过标准化）后每一行的元素作为k维欧式空间的一个数据，将这N个数据使用k-means或者其他传统的聚类方法聚类，最后将这个聚类结果映射到原始数据中（ 对矩阵N*k按每行为一个数据点，进行k-means聚类，第i行所属的簇就是原来第i个样本所属的簇），原始数据从而达到了聚类的效果。

备注：计算特征值和特征向量

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/7.png">

谱聚类算法的主要优点有：

1）谱聚类只需要数据之间的相似度矩阵，因此对于处理稀疏数据的聚类很有效。这点传统聚类算法比如K-Means很难做到

2）由于使用了降维，因此在处理高维数据聚类时的复杂度比传统聚类算法好。

谱聚类算法的主要缺点有：

1）如果最终聚类的维度非常高，则由于降维的幅度不够，谱聚类的运行速度和最后的聚类效果均不好。

2) 聚类效果依赖于相似矩阵，不同的相似矩阵得到的最终聚类效果可能很不同。

参考：https://blog.csdn.net/songbinxu/article/details/80838865

<https://blog.csdn.net/qq_24519677/article/details/82291867>

<https://www.cnblogs.com/Leo_wl/p/3156049.html>

调用API：sklearn.cluster.SpectralClustering

参考：https://www.jianshu.com/p/d35aea90ec5d

**2.图神经网络的无监督学习算法：**

（1）LINE

代码：https://github.com/tangjianpku/LINE

内容：提出算法定义了两种相似度：一阶相似度和二阶相似度，一阶相似度为直接相连的节点之间的相似性，二阶相似度为共享节点的节点间相似性。

目标：对网络节点进行表达，如n维向量表示某节点，最小化相邻节点对应向量之间的距离，同时最大化不相邻节点对应向量之间的距离。

论文举例：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/8.png">

一阶相似：

1阶相似度用于描述图中成对顶点之间的局部相似度，形式化描述为若u,v之间存在直连边，则边权wuv即为两个顶点的相似度，若不存在直连边，则1阶相似度为0。

如上图，6和7之间存在直连边，且边权较大，则认为两者相似且1阶相似度较高，而5和6之间不存在直连边，则两者间1阶相似度为0。

二阶相似：

虽然5和6之间不存在直连边，但是他们有很多相同的邻居顶点(1,2,3,4)，这其实也可以表明5和6是相似的，而2阶相似度就是用来描述这种关系的。

形式化定义为令pu=(wu,1,...,wu,∣V∣)表示顶点u与所有其他顶点间的1阶相似度，则uu与vv的2阶相似度可以通过pu和pv的相似度表示。若u与v之间不存在相同的邻居顶点，则2阶相似度为0。

参考：https://www.jianshu.com/p/79bd2ca90376

（2） LargeVis

是一种基于LINE的高维数据可视化算法

1、将高维数据表示为网络

2、使用LINE算法将网络嵌入到2维或3维空间

3、绘制出嵌入后得到的向量。

 

（3）图自编码器（Graph Auto-encoder）& VGAE （Variational Graph Auto-Encoders）

目标：链路预测

代码：https://github.com/tkipf/gae

论文：Variational Graph Auto-Encoders

特点：结合了节点特征以及网络结构信息

内容：模型分为2个子模型：（1）推理模型Inference model （2）生成模型Generative model  

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/9.png">

推理模型Inference model ---- 2层GCN

①利用邻接矩阵A(N×N)和度矩阵D(N×N)计算归一化拉普拉斯矩阵A~ 

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/10.png" width="300">

②利用特征矩阵X(N×D)和归一化拉普拉斯矩阵A~ 计算GCN(X,A)

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/11.png" width="300">

③计算均值向量矩阵µ 

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/12.png" width="300">

④计算相似对数σ

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/13.png" width="300">

⑤随机生成潜在变量矩阵Z(N×F)并更新

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/14.png">

（2）生成模型Generative model

①基于潜在变量矩阵Z更新邻接矩阵A

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/15.png">

备注：σ(·)是logistic sigmoid 函数

（3）Non-probabilistic graph auto-encoder (GAE) model

①VGAE：基于新的邻接矩阵A和特征矩阵X生成新的潜在变量矩阵Z（Embedding）

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/16.png" width="300">

参数优化：阅读文献Learning部分

数据集：https://linqs.soe.ucsc.edu/data

（3）反正则化图自编码器Adversarially Regularized Graph Autoencoder for Graph Embedding（ARGA）

& 反正则化变分自编码器 （ARVGA）

论文：Adversarially Regularized Graph Autoencoder for Graph Embedding

代码：https://github.com/Ruiqi-Hu/ARGA

内容：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/17.png">

目的：基于嵌入矩阵Z和节点信息矩阵X(node content matrix)---重构--->图A

谱卷积：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/18.png">

具体f函数：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/19.png">

其中：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/20.png">

Graph Encoder 2-versions:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/21.png">

Graph convolutional encoder:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/22.png">

Variational Graph Encoder:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/23.png">

Link prediction Layer:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/24.png">

Graph Autoencoder layer:

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/25.png">

Adversarial Model cost：

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/26.png">

Adversarial Graph Autoencoder Model

<img src="https://github.com/jm199504/Paper-Notes/blob/master/GraphTech-And-GNN-About-Unsupervised-Learning/images/27.png">

Baseline：

1. K-means is a classical method and also the foundation of many clustering algorithms.

2. Graph Encoder [Tian et al., 2014] learns graph embedding for spectral graph clustering.

3. DNGR Deep Neural Networks for Learning Graph Representations (DNGR) [Cao et al., 2016] trains a stacked denoising autoencoder for graph embedding.

4. RTM [Chang and Blei, 2009] learns the topic distributions of each document from both text and citation.

5. RMSC [Xia et al., 2014] employs a multi-view learning approach for graph clustering.

6. TADW [Yang et al., 2015] applies matrix factorization for network representation learning.

附：
**路径寻找**
1.经典算法
>搜索（DFS、BFS）
>
>最短路（Dijkstra、Floyd）
>
>最小生成树（Boruvka）
>
>进阶算法
>
>搜索（词典排序广搜、迭代深化深搜）
>
>最短路（A*启发式搜索、Yen's的K条最短路）
>
>最小生成树（Kruskal、Prime、Chu-Liu/Edmonds）

2.应用场景
>文件搜索
>
>GPS最优路径
>
>IP路由规划
>
>管道铺设
>
>其他高层图算法的基础
>
>任务分配与项目选择

**中心度衡量**
1.经典算法
>度中心度（点的度数越大越重要）
>
>接近中心度（到其他点距离越近越重要-距离衡量）
>
>中介中心度（作为其他点间的中心节点-缺失对其他节点最短路径的影响程度）
>
>特征向量中心度（全局性的中心度-周边节点构建特征）
>

2.进阶算法
>PageRank算法（谷歌搜索引擎、全局重要性）
>
>HITS（主题依赖性、局部计算、收敛更快）

3.应用场景
>网络节点排序（学术文献影响力、社交网络影响力）
>
>交通网络优化（交通网络节点重要性分析）
>
>搜索引擎优化（搜索结果排序）
>

**数据集参考文献：**

[1]: W. W. Zachary, An information flow model for conflict and fission in small groups, Journal of Anthropological Research 33, 452-473 (1977) 

[2]: M. Girvan and M. E. J. Newman, Proc. Natl. Acad. Sci. USA 99, 7821-7826 (2002). 

[3]: V.Lusseau, K .Schneider, OJ .Boisseau et al. The Bottlenose Dolphin Community of Doubtful Sound Features a Large Proportion of Long-Lasting Associations. Behavioral Ecology and Sociobiology, 2003, 54(4):392-405 

[4]: L. A. Adamic and N. Glance, “The political blogosphere and the 2004 US Election”, in Proceedings of the WWW-2005 Workshop on the Weblogging Ecosystem (2005) 

[5]: Zhou Y, Cheng H, Yu J X. Clustering large attributed graphs: An efficient incremental approach[C]//Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010: 689-698. 

[6]: Zhou Y, Cheng H, Yu J X. Graph clustering based on structural/attribute similarities[J]. Proceedings of the VLDB Endowment, 2009, 2(1): 718-729. 

[7]: Klimt B, Yang Y. Introducing the Enron Corpus[C]//CEAS. 2004. 

[8]: Yang T, Jin R, Chi Y, et al. Combining link and content for community detection: a discriminative approach[C]//Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2009: 927-936. 

[9]: Lu Q, Getoor L. Link-based classification[C]//ICML. 2003, 3: 496-503. 

[10]: Dang T A, Viennet E. Community detection based on structural and attribute similarities[C]//International Conference on Digital Society (ICDS). 2012: 7-12. 

[11]: Xirong Li, Cees G.M. Snoek, and Marcel Worring, Learning Social Tag Relevance by Neighbor Voting, in IEEE Transactions on Multimedia (T-MM), 2009 

[12]: Newman M E J. Finding community structure in networks using the eigenvectors of matrices[J]. Physical review E, 2006, 74(3): 036104.
