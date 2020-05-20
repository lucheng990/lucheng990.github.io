---
tag: 机器学习
---









Word2Vec是谷歌在2013年提出的将词汇进行distributed representation的方法。其中主要包含两套模型：CBOW和Skip-gram(s)。



这两个模型都是浅层（两层）的神经网络模型，结构比较简单，但是在实际训练的过程中会存在一个计算量大的问题，后面会解释，这是因为softmax激活函数的原因。相应的，word2vec的作者也提出了两套近似（对softmax进行一个近似）训练方法，分别是：**Negative Sampling（负采样）**和**hierarchical softmax（层次softmax）**来解决计算量大的问题。



---



先来看看模型的具体结构：



## Skip-gram(s) （跳字模型）



![a1](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic2/03Word2Vec%E4%BB%A5%E5%8F%8A%E8%B4%9F%E9%87%87%E6%A0%B7%E5%92%8C%E5%B1%82%E5%BA%8Fsoftmax/a1.png)



如上图所示，跳字模型是拿其中的一个单词（中心词）来预测周围的词（背景词）。输入的单词（1 of N encoding，向量的维度是V行1列，实际运算有转置，V是词汇表的长度）经过第一个参数\\(W_{V \times N}\\)(下标表示维度，这个V行N列的矩阵又是这个模型的look up table)得到的N维度的向量就是这个单词的**distributed representation**。然后这个向量**乘以（点乘，内积）**一个参数\\(W'_{N\times V}\\)（这个参数可以理解为是各个单词的**背景词**表示），最后得到一个向量（1行V列），这个向量经过Softmax就得到一个生成各个单词的概率分布，将这个概率分布和实际的\\(\hat{y}\\)做交叉熵就得到损失函数。





这里有几点，两个向量的点乘（内积）除以这两个向量的模的积表示余弦相似度，在这里可以直接用两个向量的积表示。词典里面的所有单词都默认有两种表示（为了简化后面的数学操作和便于理解。当然如果用一种也可以），分别是v(作为中心词)和u(作为背景词)。



最大似然：


$$
\prod_{t=1}^T \prod_{-m \le j \le m,j\ne 0} P(w^{t+j}| w^t)
$$



也就是要遍历整个文本（长度为T），对每一个中心词预测其背景词（周围词）。乘变加，可以得到损失函数：


$$
-\frac{1}{T} \sum_{t=1}^T \sum_{-m\le j\le,j\ne 0}\log{P(w^{t+j}| w^t)} \tag{0}
$$




假设输入为x（1 of N encoding)，经过第一层参数得到这个单词的distributed representation，\\(v_c\\)，下标c表示是中心词：


$$
x^TW=v_c
$$



后面还有一个参数\\(W'\\)，这个参数的形状是N行V列（看成是有V个N维度的向量，也就是单词的u表示、背景词表示）。


$$
W' = [u_1, u_2,\cdots, u_{\lvert V \rvert}]
$$


和\\(v_c\\)相乘以后，得到一个1行V列的向量（这里随便对v或者u进行转置，假设是对u进行转置的话），中间的每一项都表示两个的相似度：


$$
v_c \cdot W' = [u_1^Tv_c,u_2^Tv_c, \cdots u^T_{\lvert V\rvert}v_c] \tag{1}
$$

Skip-Gram模型的输出\\(\hat{y}\\)是取中心词附近某个窗口(window，假设窗口大小为2，则取中心词前两个和后两个单词)中的词的1 of N encoding形式，实际写损失函数和更新参数的时候都是以一个背景词为最小单位的。拿(1)式经过softmax函数求得，对于中心词\\(v_c\\)预测到背景词\\(u_o\\)的概率，也就是上面的(0)式后半部分，可以写成：


$$
P(w_o | w_c)= \frac{\exp{(u_o^T v_c)}}{\sum_{i\in V} \exp{(u_i^Tv_c)}} \tag{2}
$$


损失函数对第一个参数\\(W\\)求偏导（取关键部分，结合(0)，(1)，(2)三式），其实也就是对中心词的向量表达求偏导（因为最开始的输入是one hot形式的），得到：




$$
\frac{\partial \log P(w_o|w_c)}{\partial v_c}= u_0-\sum_{j \in V}\frac{\exp (u_j^T v_c)}{\sum_{i \in V} \exp (u_i^T v_c)} u_j \tag{3}
$$





由(2)(3)式，可得：


$$
\frac{\partial \log P(w_o|w_c)}{\partial v_c} = u_o - \sum_{j \in V}P(w_j |w_c)u_j \tag{4}
$$




由（4）式可以看出，每一次求偏微分，每一次迭代的计算都涉及到整个词典（大小为V），如果这个词典有上亿个单词，那么每一次计算量都是会很大。



---



## CBOW(连续词袋模型)





![a2](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic2/03Word2Vec%E4%BB%A5%E5%8F%8A%E8%B4%9F%E9%87%87%E6%A0%B7%E5%92%8C%E5%B1%82%E5%BA%8Fsoftmax/a2.png)





如上图所示，CBOW用周围词预测中心词。这个模型的输入可以比较灵活，比如求和然后平均等。



最大似然：


$$
\prod_{t=1}^T P(w^t | w^{t-m}, \cdots, w^{t-1}, w^{t+1},\cdots, w^{t+m})
$$


其中m为窗口大小。改为加法，损失函数如下：


$$
- \sum_{t=1}^T \log P(w^t | w^{t-m}, \cdots, w^{t-1}, w^{t+1},\cdots, w^{t+m})
$$


每次用中心词周围20个背景词来预测这个中心词，其中：


$$
P(w_c| w_{o_1},\cdots, w_{o_{2m}})= \frac{\exp [u^T_c(v_{o_1}+\cdots +v_{o_{2m}})/(2m)]}{\sum_{i \in V}\exp [u_i^T(v_{o_1}+\cdots + v_{o_{2m}})/(2m)]}
$$





由上两式求得偏微分：




$$
\frac{\partial \log P(w_c|w_{o_1},\cdots,w_{o_{2m}})}{\partial v_{o_i}} = \frac{1}{2m} (u_c - \sum_{j \in V}\frac{\exp (u_j^Tv_c)}{\sum_{i \in V} \exp (u_i^T v_c)}u_j)
$$


也就是：


$$
\frac{\partial \log P(w_c|w_{o_1},\cdots,w_{o_{2m}})}{\partial v_{o_i}} =  \frac{1}{2m}(u_c-\sum_{j \in V}P(w_j | w_c)u_j)
$$



可以看出，每一次计算也都会涉及到整个词典。





---



下面是两个近似训练方法，来减少计算量。





## Negative Sampling(负采样)



跳字模型和连续词袋模型使用softmax函数是因为需要考虑到背景词可能是词典中的任一项，所以分母计算量会特别大。负采样的思想是，考虑到背景词可能是词典中的几项（几项里面包括一个正标签），这个“几项“每次计算都会从某个分布（例如某个均匀分布）中随机采样得到。





同时，使用Negative Sampling，上面（1）式会经过一个Sigmoid函数（而不是Softmax，因为从某个分布随机采样label，所以生成整个词典中的每个单词的和的概率不必为1)，这代表这由中心词生成每个背景词的概率。



这里要注意的是，上面的随机采样是指采样的“噪声”，即除了target外的采样，每次更新迭代参数都会包含一个target以及从随机采样中得到的噪声。





假设一次迭代中采样K个噪音，那么，由中心词生成某个背景词的概率可以表示为（以跳字模型为例）：


$$
P(w_o|w_c) = P(D=1|w_o,w_c)\prod_{k=1,w_k \sim P(w)}^K P(D=0|w_k,w_c) \tag{5}
$$




上式表示，由中心词\\(w_c\\)生成背景词\\(w_o\\)的这一事件，可以看成是，中心词和target（即这个背景词）同时出现；中心词和K个噪音词\\(w_k\\)不同时出现这两件事的联合概率。这个概率P由Sigmoid函数得到，即：




$$
P(D=1|w_o,w_c) = \sigma(u_o^Tv_c)= \frac{1}{1+\exp (-u_o^Tv_c)} \tag{6}
$$

$$
P(D=0|w_k,w_c)=1-\sigma(u^T_{i_k}v_c)=1-\frac{1}{1+\exp (-u_{i_k}^Tv_c)}  \tag{7}
$$


（7）式对比（6）式加个负号并被1减，是因为：第一，两个是对立事件；第二，（5）式是要去最大化的概率，那么就要最小化（7）中的这个sigmoid函数项，所以加负号。

由(5)(6)(7)式并取log可以得到：


$$
\log P(w_o|w_c) = \log \frac{1}{1+\exp(-u_o^Tv_c)} + \sum_{k=1,w_k\sim P(w)}^K \log[1-\frac{1}{1+\exp (-u^T_{i_kv_c})}] \tag{8}
$$


加个负号，并对最右边的项进行分子有理化，得到：


$$
-\log P(w_o|w_c) = -\log\frac{1}{1+\exp (-u_o^Tv_c)}-\sum_{k=1,w_k\sim P(w)}^K \log\frac{1}{1+\exp(u^T_{i_k}v_c)} \tag{9}
$$


(9)式对比(4)式，可以看出计算量大大减少了，(9)式的计算量被K的值限制住，实做时这个K通常会比较小，例如4或者5。



连续词袋模型也同理，最后的要最小化的概率（损失函数）为：


$$
\begin{align*}
& -\log P(w^t|w^{t-m},\cdots,w^{t-1},w^{t+1},\cdots,w^{t+m}) \\
=& -\log \frac{1}{1+\exp [-u_c^T(v_{o_1}+\cdots+v_{o_{2m}})/(2m)]}-\sum_{k=1,w_k \sim P(w)}^K \log \frac{1}{1+\exp[u^T_{i_k}(v_{o_1}+\cdots+v_{o_{2m}})/(2m)]}
\end{align*}
$$





计算开销也被预设的K的值给bound住。



---



## Hierarchical Softmax（层序Softmax）



层序softmax分别将每一个\\(\hat{y}\\)作为一棵二叉树中的叶子结点，这个二叉树可以是完全二叉树（方便构建），或者是哈夫曼树（最优，计算开销最小）。对于每一个结点，假设向左为0，向右为1，模型的输出经过sigmoid函数，每一次会计算向左走还是向右走，最后到达叶子结点。



如图所示，图中的n表示路径，123表示层数：

![a3](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic2/03Word2Vec%E4%BB%A5%E5%8F%8A%E8%B4%9F%E9%87%87%E6%A0%B7%E5%92%8C%E5%B1%82%E5%BA%8Fsoftmax/a3.png)



由\\(w_i\\)生成某一个背景词的\\(w_o\\)的概率如下：



$$
P(w|w_i)= \prod_{j=1}^{L(w)-1} \sigma([n(w,j+1)=right\_child(n(w,j))]\cdot u^T_{n(w,j)}v_i)
$$





其中，中括号里面是一个判断函数，如果要走到某一个叶子结点（target），在上面的结点中，向右走为1，向左走为-1；\\(\sigma\\)为sigmoid函数，表示概率。公式里面的\\(v_i\\)在前向传播传过来。



在这个公式中，由一个中心词生成各个背景词的概率的和为1：


$$
\sum_{w=1}^{ V} P(w|w_i) = 1
$$




最后，Hierarchical softmax的计算复杂度为


$$
o(\log(\lvert V\rvert))
$$
