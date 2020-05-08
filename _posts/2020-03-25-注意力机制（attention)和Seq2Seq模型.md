---
tag: 机器学习
---







## Seq2Seq





在使用Auto Encoding的时候有用到Encoder-Decoder模型，但是在自然语言处理的很多应用中，更具体的，通常使用或者说是叫它**seq2seq**模型，常常用来处理一些输入输出都是不定长序列的（N-M）问题。这一模型的结构比较简单，但是可以有很多的灵活变形，例如，有其中的一种结构（[来源](https://zh.d2l.ai/chapter_natural-language-processing/seq2seq.html)）：



![a1](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic2/02%20Attention%E5%92%8CSeq2Seq/a1.png)



如上图所示，在一个time step中，编码器的最后一个时刻（t=T）的hidden layer输出会被当做输入投入到所有时刻（t=1—T）的解码器中。`<eos>`和`<bos>`是特殊符号，作为序列的（结尾和开始）处的输入，表示end和beginning of the sequence。



在另外有些Tutorials中，该输出仅投入到(t=1)时刻的解码器中。在这种case中，decoder的每一个hidden layer output都要同时去support 该时刻的输出和所有后面时刻的输出。所以这样的话，如果一个序列很长，但是输出的维度又不够大造成**信息表达不够充分**；而且**先输入到网络中的信息很容易被覆盖、丢失掉**，这将会对最后的输出造成一定的影响。







上面第一种（图示）的做法是要更好的，能够保留下更多有价值的信息。在这种情况下，解码器的隐藏层输出仅需支持该时刻的输出以及为下一时刻（t+1）提供一个包含上下文（context）的信息。这种做法就是Attention mechanism(注意力机制)的雏形。



---





## Attention 



上面的图示做法中，编码器的输出可以看做是一个背景变量，作为解码器的输入，但是这个背景变量是固定的(Input Recurrence，在每一时刻都recurrence的输出一遍)。



Attention的做法是，**explicitly地去train一个model**，让这个model能够自己决定在某一时刻的输入，如下图所示([来源](https://easyai.tech/ai-definition/encoder-decoder-seq2seq/))：





![a2](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic2/02%20Attention%E5%92%8CSeq2Seq/a2.png)





Encoder输出一个向量（**要求是可以break into part**)，然后attention机制对这个向量中的每一个元素进行加权平均（权重之和为1，相当于是一个概率分布函数），分别输入到解码器的各个时刻中。







下面是一个Image Caption的例子（[来源](https://www.youtube.com/watch?v=VTXgPNmENG0&list=PLlPcwHqLqJDkVO0zHMqswX1jA9Xw7OSOK&index=8&t=421s)）：





![a3](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic2/02%20Attention%E5%92%8CSeq2Seq/a3.png)





如上图，Encoder部分是一个CNN model，CNN的输出会被加权平均输入到每一个timestamp中去，那么这个**加权平均的权重（和为1，图为绿色）是怎么得到的？**



#### 权重的获得



这个权重的获得方法有很多，下面举一个例子，如图[来源](https://youtu.be/VTXgPNmENG0?list=PLlPcwHqLqJDkVO0zHMqswX1jA9Xw7OSOK&t=748)：



![a4](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic2/02%20Attention%E5%92%8CSeq2Seq/a4.png)



上一个timestamp (t-1)的hidden layer输出为\\(a^{t-1}\\)，Encoder部分的输出为\\(x^i\\)，\\(p^T\\)和\\(q^T\\)和\\(r\\)为参数（可以训练的）。（现在要拿\\(a^{t-1}\\)和\\(x\\)来算一个match score）



可以得到：


$$
z_i =  \sigma(p^T a^{t-1} + q^Tx^i + r)
$$



其中\\(\sigma\\)为激活函数。然后把\\(z_i\\)放到一个softmax函数里面去，就能得到权重\\(b_i\\)：




$$
b_i = softmax(z)_i = \frac{\exp{z_i}}{\sum_j \exp{z_j}}
$$




这里的softmax的好处是：



* 进行normalization
* 有一个exp的操作，能够把差距拉开，重要的部分更大，不重要的部分更小



得到这个权重，就可以和最开始的\\(x\\)做內积（加权平均），然后输入到这个timestamp中去。



要注意的是，**这里的p、q、r是shared，在不同的timestamp中，在不同序列中**



#### 权重的优化



这里的几个参数p、q、r可以和上面的rnn主体一起训练，因为梯度，或者说error signal可以反向传播过来。





在这里使用attention可以有一个很明显的优点：decoder的输入size变成了每一个part的size，所以减少了decoder部分输入的维度，也就是减少了decoder部分的参数的量。







---



参考资料：





[注意力机制](https://zh.d2l.ai/chapter_natural-language-processing/attention.html)

[Encoder-Decoder 和 Seq2Seq](https://easyai.tech/ai-definition/encoder-decoder-seq2seq/)

[Attention Model Intuition](https://youtu.be/SysgYptB198?list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6)

[Attention Model](https://youtu.be/quoGRI-1l0A?list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6)





视频：



<iframe width="844" height="634" src="https://www.youtube.com/embed/VTXgPNmENG0?list=PLlPcwHqLqJDkVO0zHMqswX1jA9Xw7OSOK" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>