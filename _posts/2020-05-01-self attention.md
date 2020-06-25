---
tag: 机器学习
---







## Attention到Self-attention





Self-Attention和Attention的核心思想类似，都是对某一可分块的含有一定信息的向量中的每一分块进行加权平均，得到的结果向量就会对其中的某些分块增加额外的attention或者忽略其中无关紧要的分块。



区别是Self-Attention侧重于对自身向量(self)的重构，让自己和其他的inputs有interaction（假设输入是一个时间序列）。



使用Self-Attention的好处是：

1. 可以让一个input包含整个序列(time step)的对于这个input有用的信息
2. 在1的同时可以很好并行计算（比较LSTM）







## Self-Attention的具体做法



首先对于self-attention来说，假设输入是一个含有n个input的时间序列(time step)，输出也是n个包含更多含义的output。整个过程如[下图](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)所示：



![a1](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic2/05Self-attention/a1.png)



动态过程：



![a2](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic2/05Self-attention/a2.gif)



具体的过程说明如下：



* 假设一个time step中有3个input，每个input是一个4维的向量（上图绿色）。



input默认列向量，经过转置和拼接，得到如下向量：
$$
\begin{align*}
[&\begin{bmatrix}
1&0&1&0
\end{bmatrix} \\
&\begin{bmatrix}
0&2&0&2
\end{bmatrix}  \\
&\begin{bmatrix}
1&1&1&1
\end{bmatrix}]
\end{align*}  \tag{1}
$$


* 每个time stamp的input都会衍生出三个向量，分别是Key、Query、Value。



这里的衍生，也就是把每个input向量分别乘上一组参数，分别得到这三个向量。例如要得到第一个input（转置后是1行4列向量）的Key（假设Key是3维，1行3列的），那么就乘上一个4行3列的矩阵**M**。这个矩阵**M**对于每个input都是**share weights**的，后面的几个inputs乘上这同一个矩阵，也得到相对应的Key。**实作的时候还要加上一层bias**。



得到Key的乘法：
$$
\begin{bmatrix}
1&0&1&0 \\
0&2&0&2\\
1&1&1&1
\end{bmatrix}
\cdot M =
\begin{bmatrix}
[0&1&1]\\
[4&4&0] \\
[2&3&1]

\end{bmatrix}
$$





**这里的参数矩阵M随机初始化，而且可以训练(梯度可以反向传播回来)，实际就是得到一层layer的过程（包含Bias）**









同理可以得到每个input的Query（上图红色）、Value（上图紫色）向量。



**注意：这里的Query vector和Key vector的维度必须要相同（因为后面要做element-wise product），而Value vector可以和另两个不同，最后的结果的Output vector的维度和Value vector的维度保持一致。**



* 计算Attention Scores



分别拿每个input的query对所有inputs(包括自己)的key做element-wise product（这里是一种求相似度的方法，和之前的Attention一样，求法有很多，例如可以用一个有一层hidden layer的前向神经网络），得到每个input对于所有inputs的Attention scores。



例如，在上面的例子中，**input1会有3个Attention Scores**(分别是对于input1,input2,input3的)。





那上面讲到的两种求相似度的方法，element-wise product和一层神经网络。这两种方法在Query和Key vector的维度较小的时候效果差不多，但是当Query和Key vector的维度比较大的时候，使用一层神经网络效果会更好，原因在论文中有解释，element-wise product在维度较大的情况下计算结果会增长比较快，计算出来的值往往会比较大，这样的话在后面经过Softmax函数的时候，梯度会落在比较小的地方，参数不容易被更新到。



**解决方法是，对于上面计算得到的3个Attention Scores，分别除以根号维度(假设Query和Key的维度为\\(d_k\\))。**





**这三个Attention Scores都是标量，为了让其和为1，经过一个Softmax函数，**（得到的还是一个标量，表示该input与其他所有input的相关的程度）。







* 计算Weighted Values



拿上面经过Softmax得到的Attention Scores分别乘上所有inputs的Value vector，得到Weighted Values。



**对于一个input1就会有3个Weighted Values**，维度和Value vector的维度一样



上述计算Attention的过程可以描述为：


$$
\mathrm{Attention}(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$


得到的这个东西在论文中叫Scaled Dot-Product Attention。



* 得到最后的Output1



对得到的3个Weighted Values做element-wise summation，得到最后的Output1。





* 用同样的方式得到最后的Output2和Output3





这样就完成了整个Self-Attention的计算，输入是n个input，输出也是n个input。当然输入和输出的维度可以是不同的。





---





参考资料：



[Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)



[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

