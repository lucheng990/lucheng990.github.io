---
tag: 机器学习
---


## Backpropagation



在进行Neural Network 参数优化的时候，要做的事情就是最小化\\(L\\)，而\\(L\\)是由每一笔数据的\\(l\\) summation over得到的。





$$
L(\theta)=\sum_{n=1}^Nl^n(\theta)
$$




要对L求偏微分，也就是要对每一个\\(l\\)求偏微分。


$$
\frac{\partial L(\theta)}{\partial w}=\sum_{n=1}^N\frac{\partial l^n(\theta)}{\partial w}
$$


所以，下面要做的事情，就是计算每一个小\\(l\\)对各个参数的偏微分，最后加起来就行了。



![01](https://i.loli.net/2018/12/20/5c1ae1a19a13e.png)



如图，对于红色框框中的第一个Layer中的第一个neuron来说，



![02](https://i.loli.net/2018/12/20/5c1ae1a1c3f05.png)



在进入神经元之前，假设


$$
z=x_1w_1+x_2w_2+b
$$




所以


$$
\frac{\partial l}{\partial w}=\frac{\partial l}{\partial z}\frac{\partial z}{\partial w}
$$




下面要做两步。

1. Forward pass:计算\\(\frac{\partial z}{\partial w}\\)，对于每个参数都要计算。
2. Backward pass计算\\(\frac{\partial l}{\partial z}\\)



**第一步会比较简单**


$$
\frac{\partial z}{\partial w_1}=x_1\\
\frac{\partial z}{\partial w_2}=x_2\\
$$


发现：**参数对于z的偏微分，恰好就是前面连接到该参数上的input。**



**在第二步中**,要计算z对l的偏微分,**即z对这个Loss函数的偏微分**，会比较麻烦。





假设




$$
a=\sigma(z)
$$


其中，\\(\sigma\\)，即a，是Activation Function。



所以


$$
\frac{\partial l}{\partial z}=\frac{\partial l}{\partial a}\frac{\partial a}{\partial z}
$$


假设\\(\sigma\\)是 Sigmoid Function，则


$$
\frac{\partial a}{\partial z}=\sigma'(z)
$$




而 
$$
\frac{\partial l}{\partial a}
$$
则和后面的一系列结构、函数等有关。



![03](https://i.loli.net/2018/12/20/5c1ae1a1c40ab.png)



假设下一个Layer有两个neuron



那么，根据Chain rule，


$$
\frac{\partial l}{\partial a}=\frac{\partial l}{\partial z'}\frac{\partial z'}{\partial a}+\frac{\partial l}{\partial z''}\frac{\partial z''}{\partial a}
$$


其中


$$
\frac{\partial z'}{\partial a}=w_3 \\
\frac{\partial z''}{\partial a}=w_4
$$


同时，假设后面两项已经知道，即
$$
\frac{\partial l}{\partial z'}和\frac{\partial l}{\partial z''}
$$
已知。





那么


$$
\frac{\partial l}{\partial z}=\sigma'(z)[w_3\frac{\partial l}{\partial z'}+w_4\frac{\partial l}{\partial z''}]
$$




![04](https://i.loli.net/2018/12/20/5c1ae1a1c68ea.png)



其中，方括号里的内容，又可以看成是一个另类的neuron，如下图所示，\\(w_3\\)和\\(w_4\\)分别是这个另类的neuron的参数，最后乘上一个已知的**常量\\(\sigma'(z)\\)**。

![05](https://i.loli.net/2018/12/20/5c1ae1a1c3e24.png)





现在，如果这已经是到最后一个Layer了，那么很好计算。

![06](https://i.loli.net/2018/12/20/5c1ae1a1c3e9c.png)





如果不是最后一个output layer，那么就要继续算下去，直到算到最后......







**现在从后往前来算，即从output的地方来算**，就会比较简单了。

![07](https://i.loli.net/2018/12/20/5c1ae1a1c6697.png)

如图所示的神经网络，先算


$$
\frac{\partial l}{\partial z_5}\quad和\quad \frac{\partial l}{\partial z_6}
$$


然后，再根据前面的**公式**，


$$
\frac{\partial l}{\partial z}=\sigma'(z)[w_3\frac{\partial l}{\partial z'}+w_4\frac{\partial l}{\partial z''}]
$$


这样一直往前算。



![08](https://i.loli.net/2018/12/20/5c1ae1a1c673f.png)



就能把所有的偏微分算出来了。

![09](https://i.loli.net/2018/12/20/5c1ae1a23d42b.png)



