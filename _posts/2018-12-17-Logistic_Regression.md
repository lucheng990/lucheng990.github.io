---
tag: 机器学习
---


# Logistic Regression

## Model



由前面Classification中用的model可以定义一组**Function Set**


$$
P_{w,b}(C_1|x)=\sigma(z)=f_{w,b}(x)
$$


其中，


$$
z=w\cdot x+b=\sum_i w_ix_i +b\\
\sigma(z)=\frac{1}{1+exp(-z)}
$$


用图像来描述：

![01](https://i.loli.net/2018/12/17/5c17b45c70b36.png)

红色框框部分就是定义的**Function Set**。通过Sigmoid Function得到output，output值介于0到1之间。



这一过程就是Logistic Regression。



## Goodness of a Function

![02GoodnessOfaFunction](https://i.loli.net/2018/12/17/5c17b45cd0c4a.png)

1. 首先要有一组**training data**，x为input，output为其所属的分类。
2. 假定这些data是基于某个几率分布function生成的。
3. 定义Likelihood Function : 对于每一个类别来说


$$
L(w,b)=f_{w,b}(x^1)f_{w,b}(x^2)(1-f_{w,b}(x^3))...f_{w,b}(x^N)
$$




**注意：对于二分类问题来说，因为x^3属于第二类，而且这里的这个function 是结果为第一类的概率，所以应该是用
$$
1-f_{w,b}(x^3)
$$
来作为因子**



能够最大化L的w,b就是要找的最好的w,b。但是，这里有很多东西相乘所以不是太方便，可以做一个数学上的技巧进行一定的变换。**即对Likelihood函数（整体）取负对数**。


$$
w^*,b^*=arg\;\mathop{max}\limits_{w,b}L(w,b)
$$


就变成了


$$
w^*,b^*=arg\;\mathop{min}\limits_{w,b}-\ln L(w,b)
$$


这样的话，Likelihood函数就变成了加法项。还可以对每一个y^hat定一个值，在Likelihood 函数中也做相应变换，得到一个每一个项结构相近的Likelihood 函数公式 。

![03change](https://i.loli.net/2018/12/17/5c17b45d253d7.png)



这样一来，就方便写一个更加简便的公式，便于接下来的处理。

![04](https://i.loli.net/2018/12/17/5c17b45d49b89.png)

这样，原来的Likelihood函数就变成了


$$
-\ln L(w,b)=\sum_{n=1}^{n} -[\hat{y}^n\ln f_{w,b}(x^n)+(1-\hat{y}^n)\ln (1-f_{w,b}(x^n)]
$$


这个函数的值越小越好



其中，中括号里面的项的sum是两个伯努利分布的 Cross entropy ,交叉熵。



>  交叉熵公式如下：


$$
H(p,q)=-\sum_x p(x)\ln (q(x))
$$


其中，p和q为两个关于x的分布函数。交叉熵代表了这两个Distribution有多接近。



## Find the best Function

接下来要做的事情，就是要找到一个w和b，使得含有Likelihood 的这一项最小。



方法可以用**Gradient Descent**



$$
-\ln L(w,b)=\sum_{n=1}^{n} -[\hat{y}^n\ln f_{w,b}(x^n)+(1-\hat{y}^n)\ln (1-f_{w,b}(x^n)]
$$

$$
\frac{\partial{-\ln L(w,b)}}{\partial w_i}=\sum_n-[\hat{y}^n\frac{\partial \ln f_{w,b}(x^n)}{\partial w_i}+(1-\hat{y}^n)\frac{\partial\ln (1-f_{w,b}(x^n))}{\partial w_i}]
$$


其中


$$
f_{w,b}(x)=\sigma(z)=\frac{1}{1+e^{-z}} \qquad , z=w\cdot x+b=\sum_iw_ix_i+b
$$


根据链式求导法则


$$
\frac{\partial \ln f_{w,b}(x^n)}{\partial w_i}=\frac{\partial \ln f_{w,b}(x)}{\partial z}\frac{\partial z}{\partial w_i} \qquad ,\frac{\partial z}{\partial w_i}=x_i \\
\begin{align*}
&而\quad \frac{\partial \ln f_{w,b}(x)}{\partial z}=\frac{\partial \ln\sigma(z)}{\partial z}=\frac{1}{\sigma(z)}\cdot \frac{\partial\sigma(z)}{\partial z}=\frac{1}{\sigma(z)}\cdot \sigma(z)\cdot(1-\sigma(z)) \\
&=(1-\sigma(z))=(1-f_{w,b}(x^n))
\end{align*}
$$


所以


$$
\frac{\partial \ln f_{w,b}(x^n)}{\partial w_i}=(1-f_{w,b}(x^n))\cdot x_i^n
$$








同理


$$
\frac{\partial\ln (1-f_{w,b}(x^n))}{\partial w_i}=-f_{w,b}(x^n)\cdot x_i^n
$$


所以最后结果：


$$
\frac{\partial{-\ln L(w,b)}}{\partial w_i}=\sum_n-[\hat{y}^n(1-f_{w,b}(x^n))\cdot x_i^n-(1-\hat{y}^n)f_{w,b}(x^n)\cdot x_i^n]
$$




将中括号里的展开，并把
$$
x_i^n
$$
提到外面来，可以化简：



![05simplify](https://i.loli.net/2018/12/17/5c17b45c2674c.png)



所以


$$
\frac{\partial{-\ln L(w,b)}}{\partial w_i}=\sum_n-(\hat{y}^n-f_{w,b}(x^n))\cdot x_i^n
$$


在**Gradient Descent** 更新参数的时候


$$
w_i \gets w_i-\eta \cdot \sum_n-(\hat{y}^n-f_{w,b}(x^n))\cdot x_i^n
$$


可见，参数的更新步伐取决于：

* Learning Rate，自己调节
* \\(x_i^n\\)，即input，来自于Data
* 小括号里面的项，这个表示output 和理想的目标的差距大小。其值越大表示更新的步伐需要越大。



神奇的地方在于，这个公式和Linear Regression的参数更新公式一模一样

![06thesame](https://i.loli.net/2018/12/17/5c17b45abe150.png)



只不过，在Logistic Regression里面的\\(\hat{y}^n\\)只能是 0或者1；而Linear Regression里面可以使任何数。



###  在Logistic Regression 第二步Goodness of the function里面不能用 Square Error的原因分析



![07cannot](https://i.loli.net/2018/12/17/5c17b45d6285a.png)

![08cannot2](https://i.loli.net/2018/12/17/5c17b45d5c56b.png)

定义 Square Error以后，在第三步进行优化的时候。


$$
\frac{\partial (f_{w,b}(x)-\hat{y})^2}{\partial w_i}=2\cdot(f_{w,b}(x)-\hat{y})\cdot f_{w,b}(x)\cdot (1-f_{w,b}(x))\cdot x_i
$$


现在假定\\(\hat{y}^n\\)为1，即第一类。发现不管其算出来的结果是1还是0，偏微分都为0(都会有一项是0)。这点不符合要求。



在\\(\hat{y}^n\\)为0时候也一样。



更直观地表示：

![09visualize](https://i.loli.net/2018/12/17/5c17b45dcab9d.png)



如图，黑色为使用Cross Entropy 优化时候的图，红色为使用 Square Error 优化时的图。



红色的面，在距离目标很远的时候，微分值很小；距离目标比较近的时候微分值也比较小。



## Discriminative和Generative的比较

* 两者的Model (Function set) 是一样的（前提是 Generative中 share \\(\Sigma\\))。




$$
P(C_1|x)=\sigma(w\cdot x+b)
$$


* 在**Discriminative**中，直接设w和b，并通过一系列手段来进行参数优化。
* 在**Generative**中，要做的是找到\\(\mu^1\\)和\\(\mu^2\\)和\\(\Sigma^{-1}\\)。然后得到：


$$
w^T=(\mu^1-\mu^2)^T\Sigma^{-1} \\
b=-\frac{1}{2}(\mu^1)^T(\Sigma^1)^{-1}\mu^1+\frac{1}{2}(\mu^2)^T(\Sigma^2)^{-1}\mu^2+\ln \frac{N_1}{N_2}
$$




## 两种方法得到的w和b不相同

通常认为Discriminative model 的 perfomance 会更好。



---

> e.g.



一个二元分类问题

![10example](https://i.loli.net/2018/12/17/5c17b45b19523.png)

input 有两个feature， 第一类有一笔数据，第二类有12笔数据。（类似与门）



testing data 有一组数据，要求给出这是属于哪一类。



### Generative Model


$$
P(C_1|x)=\frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}
$$


直接使用伯努利的distribution（input的两个feature的取值只有0或者是1）



因为testing data数据全是1，所以只要算出竖线左边为1的条件下的概率。其中


$$
P(C_1)=\frac{1}{13}\\
P(C_2)=\frac{12}{13}\\
P(x_1=1|C_1)=1\\
P(x_2=1|C_1)=1\\
P(x_1=1|C_2)=\frac{1}{3}\\
P(X_2=1|C_2)=\frac{1}{3}
$$


带入上面的公式，最终得到

![11result](https://i.loli.net/2018/12/17/5c17b5765b480.png)


$$
P(C_1|x)<0.5
$$


显然是错误的。



原因：

1. 在使用朴素贝叶斯以及数据处理的过程中，默认输入的两个feature的产生是相互独立的，但是实际上是不独立的。
2. 在朴素贝叶斯中，第二类两个参数都是1发生的几率并不为0。
3. 第二类的权重（样本数目多），样本不平衡。



### Discriminative model 就可以很好地解决



---



![12versus](https://i.loli.net/2018/12/17/5c17b57bbf85b.png)



![13compare](https://i.loli.net/2018/12/17/5c17b583eb9f5.png)



## Multi-class Classification



> Softmax 函数：在数学，尤其是概率论和相关领域中，Softmax函数，或称归一化指数函数，是逻辑函数的一种推广。它能将一个含任意实数的K维向量 “压缩”到另一个K维实向量 中，使得每一个元素的范围都在  之间，并且所有元素的和为1。

![14softmax](https://i.loli.net/2018/12/17/5c17b57a7155e.png)

![15dflwt](https://i.loli.net/2018/12/17/5c17b584524e5.png)





多分类问题，类似 Discriminative model，从输入到输出经过Softmax函数，最后的y就是output。这个output可以当做几率来看，接下来要做的事情就是要找到最好的参数 w和b（多个）。



如果要从高斯分布推导出来，也要让 Covariance matrix 共用。



### Loss Function

**Cross Entropy**



当x属于Class1的时候，
$$
\hat{y}=
\begin{bmatrix}
1\\
0\\
0\\
\end{bmatrix}
$$




当x属于Class2的时候，
$$
\hat{y}=
\begin{bmatrix}
0\\
1\\
0\\
\end{bmatrix}
$$


当x属于Class3的时候，
$$
\hat{y}=
\begin{bmatrix}
0\\
0\\
1\\
\end{bmatrix}
$$




计算 Cross Entropy:


$$
-\sum_{i=1}^3\hat{y}_i\ln y_i
$$


这个值越小越好。







## Limitation of Logistic Regression



假设有如下例子：

![16exa](https://i.loli.net/2018/12/17/5c17b584504c3.png)



左下角的是input。这是一个二分类(Binary)问题。



如果用Logistic Regression来做，input 中的每个Feature 分别乘上\\(w_i\\)然后加上\\(b_i\\)，得到z。(这里将两个加起来了)。再通过Sigmoid 函数，得到y，如果y>=0.5或者是z>=0，可以得到第一类，否则就是第二类。



这是不能够实现的，因为在右下角的图中，不能画出一条直线来进行划分（红蓝）。



### 解决方法：Feature transformation



重新定义Feature。例如可以将原来的Feature重新定义为 到（0,0）的距离和到（1,1）的距离。

![17ft](https://i.loli.net/2018/12/17/5c17b58424684.png)



这件事情可以看成是很多个Logistic Regression 相叠加的结果。

![18](https://i.loli.net/2018/12/17/5c17b582534c8.png)





如图，蓝色的圈圈可以看成是input \\(x_1\\)和\\(x_2\\)，output为\\(x_1^\prime\\)的一个Logistic regression；绿色的圈圈看成是input \\(x_1\\)和\\(x_2\\)，output为\\(x_2^\prime\\)的一个Logistic Regression



红色圈圈的作用是来做Classification，蓝色和绿色的作用是用来做**Feature transformation**



下面来举例证明这件事情。

![19](https://i.loli.net/2018/12/17/5c17b5858e9ea.png)

![20](https://i.loli.net/2018/12/17/5c17b6104fe27.png)







将logistic regression 串接起来，最后来一个尾刀（红色圈圈），就成了神经网络——**Deep Learning**。



蓝色和绿色的圈圈中的参数（前几个Logistic Regression的参数）可以一起学习。和红色圈圈里的参数一起学习（jointly learned）。



把每一个圈圈称为一个Neuron，多个圈圈连接成Neuron net。

