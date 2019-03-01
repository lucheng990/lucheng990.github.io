# 改进 DL



当某一个Neural Network 的效果不是很好的时候，可以对整个框架进行一个修改。



训练完一个神经网络，首先要做的就是检查这个模型在Training Data 上的表现。因为不知道训练出来的模型是好是坏（可能卡在local minima)。



如果在Training Data 上的表现也不好，那么就要回头去看看Deep Learning的三个步骤，看在哪个步骤上可以进行一些改进，使得在Training Data上得到好的结果。



如果Training Data上得到了好的结果，那么可以去看看在 Testing Data上的表现。如果Testing Data上的表现不好，那么就变成了**Overfitting**，也要回到三个步骤去看。如果Testing Data上的表现也比较好就可以拿来用了。



![01](https://wx4.sinaimg.cn/large/8f3e11fcgy1g0nm8v0u2uj20lh0g0gop.jpg)





神经网络的表现不好不一定是**Overfitting**，例如，

![02](https://ws4.sinaimg.cn/large/8f3e11fcgy1g0nm9ljledj20kt0a3mz5.jpg)

如右图，横轴表示训练次数。56层（参数多）的error 比较大，20层的比较小。但这并不一定是因为Overfitting的问题。



要断定Overfitting需要检查Training Set上的表现,如左图。56层的error就已经比20层的大了。这说明这个network在一开始就没有训练好。



如果说56层的网络前20层都和20层的一样，后面都是identity，那么两者的表现应该一致。但是事实上是不一样的，56层的有可能卡在local minima。





**在文献中提出的方法，有的是要让training data 上的表现变好，有的是要让testing data上的表现变好。**







## 改进Training data上的performance



### 1.换Activation function



在06年以前，如果层数叠的太多，例如在手写数字识别这个例子中，层数太多会有比较差的表现**（利用Sigmoid函数）**



![03](https://wx2.sinaimg.cn/large/8f3e11fcgy1g0nm9z40c2j20ip0c6q77.jpg)



**准确率变差的原因并不是Overfitting，而是称作Vanishing Gradient Problem**



![04](https://wx4.sinaimg.cn/large/8f3e11fcgy1g0nma5xyngj20nk0gw0v5.jpg)



如图所示，通过反向传播来训练参数的时候（使用Sigmoid函数），靠近Input地方的参数会有很小的偏微分，学习速率非常小；而靠近output的地方的偏微分的值都会比较大，会有比较大的学习速率。当前面的参数还几乎是初始值的时候，后面已经学习完停下来了，饱和（梯度消失）。



#### 产生这一现象过的原因和Sigmoid函数的形状有关。



![05](https://ws2.sinaimg.cn/large/8f3e11fcgy1g0nmafdfcvj20mv0gi0v1.jpg)



**实际上在求某个参数对Loss的偏微分的时候，只要对该参数进行一个小小的变化，观察对Loss的影响有多大，就能估算出其偏微分大小**



**而Sigmoid Function 会将一个数（自变量）的变化从正负无穷大之间压缩到0和1之间。**



前面的参数有变化，通过后面每一个Neuron的Sigmoid 函数之后，影响都会衰减。越靠近前面的参数，衰减越大。



对于这个问题，可以通过设置dynamic learning rate来解决，另外，如果直接更换Activation Function可以更加干脆地解决。



#### ReLU(Rectified Linear Unit)



![06](https://ws2.sinaimg.cn/large/8f3e11fcgy1g0nmbn4gylj20mk0fqjsi.jpg)



ReLU函数在input 小于0的时候output是0；input大于0的时候就是其本身。



有一些理由：

1. 计算速度很快（比Sigmoid快）。
2. 有生物学上的理由。
3. 可以看成是无限个Sigmoid Function的叠加。
4. 最重要的是解决了Vanishing gradient problem。



> e.g.



![07](https://ws1.sinaimg.cn/large/8f3e11fcgy1g0nmcdt059j20n50gnmzg.jpg)



假设有一个两层的neural network。Activation function 全部选择ReLU。有一些输入小于0的作用在ReLU的operation region上以后输出为0。输出为0的对后面没有任何影响，所以可以把它们全部去掉，视同于不存在。





![08](https://ws1.sinaimg.cn/large/8f3e11fcgy1g0nmclk923j20n50hjq4p.jpg)



**对于每一层来说，输出对于输出之间都是线性的。这样一来靠近Input的参数的gradient就不会明显特别小。**





**但是对于整个神经网络来说，不是线性的，因为对于不同的输入会导致不同的神经元（单元）激活，而不是针对所有输入都是固定神经元激活**



**对于所有的输入来说，可以看成是很多分段函数，只有在很小的邻域内是线性的，远一点的输入就不是线性的了。**





#### ReLU还有很多的变形



![09](https://ws2.sinaimg.cn/large/8f3e11fcgy1g0nmcqxibzj20nn0hnq3r.jpg)



- Leaky ReLU：input 小于0的部分乘上一个0.01。
- Parametric ReLU: input小于0的部分乘上一个\\(\alpha\\)。



> 还有人提出来一个**ELU(Exponential Linear Unit)**，即在input 小于0的部分是线性的，大于0的部分非线性的。





### Maxout--Learnable activation funciton

Maxout 可以自动learn activation function



> e.g.

![q01p13](https://wx2.sinaimg.cn/large/8f3e11fcgy1g0nmcymzerj20nc0hfjtx.jpg)

如图所示，input \\(x_1\\)和\\(x_2\\)分别乘上四组参数以后得到4个不同的数。在一般的neural network里面，这四个数会分别通过activation function。

但是在maxout里面，要做的是对这四个数进行分组，例如5和7一组，-1和1一组。然后比较大小，输出较大的那个数，与此同时，也会减少第一层hidden layer的output维度。

后面的同理进行，需要的参数的组数是一般的neural network的两倍。

**Maxout中怎么分组，每一组有几个可以自己调整**

**Relu是Maxout的一种特殊情况**

![q02p14](https://wx1.sinaimg.cn/large/8f3e11fcgy1g0nmd80bd7j20nh0h2q4o.jpg)

Relu相当于每一个z加上同组的0，然后通过maxout来比较大小得到的。

#### Maxout如果给他加上适当的参数，可以变成很多其他的activation funciton

![q03p15](https://ws1.sinaimg.cn/large/8f3e11fcgy1g0nmdfvkv9j20nt0hgmz6.jpg)

如果新增的参数不是0，而是其他的，例如\\(w'\\)和\\(b'\\)，那么这个activation function就是\\(wx+b\\)和\\(w'x+b'\\)的一个取大函数。



**当然Maxout并不能learn出所有可能的activation function。**

**Maxout可以learn出的activation function都是 piecewise linear convex function(国内外对凹凸的定义不同)**

**有几段取决于一组中有多少个参数**



![q04p16](https://wx4.sinaimg.cn/large/8f3e11fcgy1g0nmdli0iaj20n70h1ta7.jpg)

#### Maxout的train法

因为Maxout是一个分段函数，有一个max的operation，所以无法整体求偏微分。

但是Maxout是可以训练的。

- 对于每一个input，去掉小的那些，剩下的部分就是一个linear model。
- 对于不同的input，去掉的小的部分都不尽相同，所以当input很多的时候，每一个参数都可以训练到。



### Adaptive Learning Rate

Adagrad是用一次微分的值来估计二次微分的值。这在PM2.5预测中loss function是一个convex function，二次微分的值是比较固定的，可以很好的估计出来，但是实际上的Error Surface可以是非常复杂的，而且维度太高并不能够提前预知。

![q05p20](https://ws3.sinaimg.cn/large/8f3e11fcgy1g0nmdrks2mj20nl0h7dl7.jpg)

![q06p21](https://wx1.sinaimg.cn/large/8f3e11fcgy1g0nmdw54vtj20ng0gnacl.jpg)

在不同的地方需要不同的learning rate。

### RMSProp

一个简单有效的方法，更新状态如下：

![q07p22](https://wx4.sinaimg.cn/large/8f3e11fcgy1g0nme0hfv5j20nl0hg3zz.jpg)

每一个\\(\sigma\\)需要知道当前的微分和前一个的微分。

RMSProp是Adagrad的一个小的变形。

### Momentum

在优化参数的时候，会遇到一些问题，例如：

- Very slow at th plateau
- Stuck at saddle point
- Stuck at local minima

有些人想到了物理世界的滚球，将惯性这个概念加到Gradient Descent

![q08p24](https://wx1.sinaimg.cn/large/8f3e11fcgy1g0nme6hec1j20mp0gh756.jpg)

![q09p28](https://ws1.sinaimg.cn/large/8f3e11fcgy1g0nmebg9ukj20nf0gwtar.jpg)

红色是偏微分说代表方向，绿色代表惯性，蓝色是真正走的方向。

**Vanilla Gradient Descent**
![q10p25](https://ws1.sinaimg.cn/large/8f3e11fcgy1g0nmeisugwj20n00g5wg9.jpg)

**Momentum**

![q11p26](https://wx4.sinaimg.cn/large/8f3e11fcgy1g0nmeng55wj20nq0hdtbm.jpg)





只考虑前一次的惯性，这里的momentum和微分无关，是指\\(\lambda v^0\\)，即前一次的movement乘上\\(\lambda\\)。

之所以只考虑前一次，是因为前一次的就包含了过去所有的gradient的总和（代入法）。

### Adam

将RMSProp和Momentum结合起来，就是Adam。

![q12p29](https://wx3.sinaimg.cn/large/8f3e11fcgy1g0nmeuirajj20np0hlgs3.jpg)

## Overfitting 的解决方法

### Early Stopping

![q13p31](https://wx4.sinaimg.cn/large/8f3e11fcgy1g0nmf0e0wrj20nh0grmyq.jpg)

在训练参数的时候，如果Epochs太多，会在Training Set上正确率训练的太好，而在Testing Data上的表现会变差。通常情况下是如上图所示，所以要在中间某个地方停下，来确保Testing set上的正确率。

**增加Validation set(验证集)，每个Epoch以后算一遍验证集上的loss value，当其不再下降即停止。**





### Regularization

之前讲过，我们更加期望一条曲线是趋于平滑的，抑制剧烈的变化。尽量减少参数的norm。要做到这一点，可以加上一项：正则项。

#### L2 regularization

> 以two norm为例

$$
L'(\theta)=L(\theta)+\lambda \frac{1}{2} \lVert \theta \rVert_2
$$

其中,
$$
L(\theta)
$$
是原始的loss，
而
$$
\lVert \theta \rVert_2 = (w_1)^2+(w_2)^2+...
$$
是新增的regularization，这里称为L2 regularization，通常不考虑bias。

![c1p33](https://wx1.sinaimg.cn/large/8f3e11fcgy1g0nmf7hsmnj20my0gpmyv.jpg)



做偏微分，进行参数更新：


$$
\frac{\partial L'}{\partial w}=\frac{\partial L}{\partial w}+\lambda w
$$

$$
w^{t+1} \gets w^t- \eta \frac{\partial L'}{\partial w}=w^t -\eta (\frac{\partial L}{\partial w}+\lambda w^t) \\
=(1-\eta \lambda)w^t - \eta \frac{\partial L}{\partial w}
$$



如果不加Regularization，那么最后参数更新：


$$
w^{t+1} \gets w^t-\eta \frac{\partial L}{\partial w}
$$


两者相比，多了一项
$$
(1-\eta \lambda)
$$
在实际处理问题的时候，常常把
$$
\eta \lambda
$$
设置为接近1的一个数，比如说0.99，使
$$
(1-\eta \lambda)
$$
接近于0。这样会使参数偏小。



**这样并不会使最后参数全部变成0，因为有后面的一项，最后两者会形成平衡**

**L2 regularization 也称为 weight decay**

#### L1 regularization

L1 regularization 会和L2 regularization 有一些小差别

**L1 regularization**


$$
\lVert \theta \rVert_1 =\lvert w_1 \rvert +\lvert w_2 \rvert +...
$$


新的Loss Function：


$$
L'(\theta)=L(\theta)+\lambda \frac{1}{2} \lVert \theta \rVert_1
$$


做偏微分：


$$
\frac{\partial L'}{\partial w}=\frac{\partial L}{\partial w} + \lambda \mathrm{sgn} (w)
$$


其中，sgn是符号函数，如果\\(w\\)是正的就是1，如果\\(w\\)是负的就是-1 。

更新参数：
$$
w^{t+1} \gets w^t -\eta \frac{\partial L'}{\partial w} =w^t-\eta (\frac{\partial L}{\partial w}+ \lambda \mathrm{sgn}(w^t)) \\
= w^t -\eta \frac{\partial L}{\partial w} -\eta \lambda \mathrm{sgn}(w^t)
$$


其中的
$$
-\eta \lambda \mathrm{sgn}(w^t)
$$
这一项，使得参数每次都向0的地方移动。道理目的和L2 regularization 类似。

**L1和L2的不同之处在于**

- L2对于参数大的向0的方向移动更快。
- L1则一视同仁，参数大小对于移动速度无关。
- 使用L1会使参数的差距比较大，有的接近0有的却很大。



### Dropout

![c2p38](https://wx2.sinaimg.cn/large/8f3e11fcgy1g0nmfeacd9j20n50fwwg7.jpg)

- 每一次update参数之前，对每一个neuron做sampling。每个nuron有p%几率被抽到然后drop掉，与该neuron相连的线都去掉，即发生了结构的改变。会变得更瘦。
- 对剩余的结构做training
- 每一次mini-batch都要重新sample出一些neuron。



**注意**

**1. 在做testing的时候没有dropout**

**2. 如果dropout rate 是 p%，那么在testing的时候，所有的参数权重都要乘上(1-p%)。**否则会有输出过大的情况。

![c3p42](https://wx3.sinaimg.cn/large/8f3e11fcgy1g0nmfk423fj20mn0gj41f.jpg)









#### 原理分析



Dropout is a kind of ensemble。

在training set 里面sample出4个Set。可以是分别有不同的结构。

![c4p44](https://wx4.sinaimg.cn/large/8f3e11fcgy1g0nmfp9r9tj20n20f0gn5.jpg)

如果将4个model取平均，那么将会得到一个更加正确合理的结果。

![c5p45](https://ws2.sinaimg.cn/large/8f3e11fcgy1g0nmftwlzoj20n60glq46.jpg)

![c6p46](https://ws4.sinaimg.cn/large/8f3e11fcgy1g0nmg2zffkj20nf0h6acq.jpg)

训练的时候，给每个model以某个mini-batch。因为这些model里面有些参数是共用的(shared)，所以不会存在参数过少的情况。

在testing的时候，理想情况是将某个input在4个model下的结果取平均，然后再输出。

在dropout里面直接用乘上(1-p%)来近似代替。





**Dropout在linear model 情况下表现会特别好。**

