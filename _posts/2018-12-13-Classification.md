---
tag: 机器学习
---

## Classification（分类问题）

> 应用举例：预测POKEMON 的不同属性（系别）



input: POKEMON (数值化表示)。



输入的是一组向量来表示各个维度的特征值，带有一个标签\\(\hat{y}\\)，用来表示正确的类别。



output: 类别



### 如果用Regression的方法来做，例如

设\\(\hat{y}\\)分别为1、2、3......，然后在测试（应用）的时候，Function 输出的值靠近哪个\\(\hat{y}\\)就定义为哪个类别。

![01RegressionCase](https://i.loli.net/2018/12/13/5c127416dec76.png)



假设input有两个属性，定为1、-1，而且Model是一条直线，那么理想的情况应该是将output 处处为0的线作为分界线，如左图绿色分界线。



但是，实际上这样做的话会出现一点问题，如右图。



1. 当出现一些蓝色的点，其结果远远大于1的值，如果继续使用Regression的方法优化下去的话，绿色线会顺时针旋转来照顾到右下角那些点，变成紫色的线，但是显然这是错的。Regression 会惩罚“太正确”的点。

2. 考虑多分类问题，如果有三个以上Class，分别定义为1、2、3。这里暗含着Class1和Class2有着某种关系，Class2和Class3有着某种关系，但是实际上可能是没有的。这样也非常不合理。





### 有一种理想的替代方式

![02Alternative](https://i.loli.net/2018/12/13/5c1272b63250a.png)

在Model里面内建一个function,g(x)。如果g(x)>0输出Class1,g(x)<0输出 Class2。



损失函数：错误估计的次数



优化方法：e.g. 

> Perceptron

> SVM



### 用概率的方法解

![03贝叶斯](https://i.loli.net/2018/12/13/5c1272a75439b.png)

> 贝叶斯公式



如果取出的是一个蓝球，那么这个球从Box1中取出的概率为：


$$
P(\mathrm{B_1}|\mathrm{Blue})=\frac{P(\mathrm{Blue|B_1})P({\mathrm{B_1}})}{P(\mathrm{Blue|B_1})P(\mathrm{B_1})+P(\mathrm{Blue|B_2})P(\mathrm{B_2})}
$$




现在把盒子换成分类（类别1、类别2）

![04贝叶斯2](https://i.loli.net/2018/12/13/5c1272af8835c.png)

![05贝叶斯3](https://i.loli.net/2018/12/13/5c1272adefde3.png)

给定一个向量X(含有各个维度信息),要知道它是从哪一个Class里面出来的几率，就需要知道如下信息：

1. Class1里面取到X的几率$$P(x|C_1)$$。
2. Class2里面取到X的几率$$P(x|C_2)$$。
3. 取到Class1本身的几率$$P(C_1)$$。
4. 取到Class2本身的几率$$P(C_2)$$。



从训练集中可以很快求出3: \\(P(C_1)\\)和4:\\(P(C_2)\\)。分别是水系的和普通系的数量除以总数。



但是，如果遇到一个不曾见过的POKEMON，即输入任意一组向量\\(x\\)，并且要知道它在Class1里面出现的几率，这和简单的红球蓝球不同。可以用到高斯分布（正态分布）来求。

![06gaosifb](https://i.loli.net/2018/12/13/5c1272bc395f7.png)

假设input 的x有2 个属性，则依据二维高斯分布可以求出概率。

![07gaosifb2](https://i.loli.net/2018/12/13/5c1272bd591aa.png)

现在要求两个参数：\\(\mu\\)和\\(\Sigma\\)

1. \\(\mu\\)：平均数，是一个二维列向量（矩阵）
2. \\(\Sigma\\)：协方差矩阵，2*2



同样的\\(\Sigma\\)不同的\\(\mu\\)；同样的\\(\mu\\)不同的\\(\Sigma\\)，得到的分布不同

![08gaosifb3](https://i.loli.net/2018/12/13/5c1272bde13c5.png)

![09gaosifb4](https://i.loli.net/2018/12/13/5c1272b42b1ad.png)

知道了这两个参数就可以求出一个新的点的分布概率。



### 求\\(\mu\\)和\\(\Sigma\\)。即求一个最好的高斯分布函数

![10qiugaosifb1](https://i.loli.net/2018/12/13/5c1272b506f06.png)

极大似然估计方法：已知某个高斯分布。这个分布的 Likelihood,记为\\(L(\mu,\Sigma)\\)。要注意这里的L 不是Loss Function。这个Likelihood就是这个高斯分布 sample 出这些点的几率，其值越大越好。


$$
L(\mu,\Sigma)=f_{\mu,\Sigma}(x^1)f_{\mu,\Sigma}(x^2)f_{\mu,\Sigma}(x^3)......f_{\mu,\Sigma}(x^{79})
$$




#### 要求出这个 高斯分布

即求出一个让Likelihood 最大的高斯分布函数，也就是要求出 $$\mu^*$$和$$\Sigma^*$$。



方法：穷举所有的\\(\mu\\)和\\(\Sigma\\)，可以用微积分来解


$$
\mu^*,\Sigma^*=arg\,\mathop{max}\limits_{\mu,\Sigma}L(\mu,\Sigma)
$$
可以得到结果：



$$
\mu^*=\frac{1}{79}\sum^{79}_{\mathrm{n=1}}x^n
$$

$$
\Sigma^*=\frac{1}{79}\sum^{79}_{\mathrm{n=1}}(x^n-\mu^*)(x^n-\mu^*)^T
$$

![11qiugaosifb2](https://i.loli.net/2018/12/13/5c1272aff029a.png)





#### 在这个例子中，可以求出高斯分布

![12qiugaosifb3](https://i.loli.net/2018/12/13/5c127376e4e45.png)



### 用已知的高斯分布函数来求出每一个类别的可能性

![13qiuknx](https://i.loli.net/2018/12/13/5c127385a9fb6.png)

![14qiuknxjg](https://i.loli.net/2018/12/13/5c127387ac38f.png)

可以发现效果并不是特别好。



而机器学习厉害的地方在于，可以处理更高纬的东西。虽然这里就算是在更高维度下也不是很好。



### 优化方法

在模型中不常看到给每个高斯函数分配自己的不同的\\(\Sigma\\)。



model参数多，更容易 overfitting



所以可以减少参数(parameter)：在设计模型时，可以故意给不同的Class 共享一个\\(\Sigma\\)。

![15yh](https://i.loli.net/2018/12/13/5c12738626e40.png)

![16yh](https://i.loli.net/2018/12/13/5c1273862675c.png)

这样的话，在优化参数的时候，Likelihood 也需要有相应的改变。不同的 Class 使用同一个 Likelihood 。



其中的结果是，\\(\mu^1\\)和\\(\mu^2\\) 和之前的算法一样，而\\(\Sigma\\)则有不同，要同时考虑两个Class。用element的数目来作为权重。


$$
\Sigma=\frac{79}{140}\Sigma^1+\frac{61}{140}\Sigma^2
$$


> Ref:Bishop,chapter 4.2.2



### 优化结果

![17yhjg](https://i.loli.net/2018/12/13/5c12738669cfc.png)

共用同一个 covariance matrix 以后，boundary从一条曲线变成了一条直线。这样的model也可以称之为linear model。



共用同一个 covariance matrix，并用上所有的参数，可以发现 正确率从54%上升到了 73%。



### 回顾



#### 三个步骤

![18threesteps](https://i.loli.net/2018/12/13/5c127386258b6.png)



1. 在第一步中的模型中，$$P(X|C_1)$$ 和$$P(X|C_2)$$ 由自己选择设计的高斯分布函数决定。在这里作为Model的参数。
2. function的好坏用 Likelihood 形容，其越大越好。\\(\mu\\)和 \\(\Sigma\\)的取值可以自己决定。





#### 为什么选择高斯分布几率模型

> 可以用其他的



#### 有一种假设

![19jieshi](https://i.loli.net/2018/12/13/5c127382dc3f7.png)

假设一个input X里面有K个互相独立的维度的参数，那么这个X在\\(C_1\\)里面被sample出来的几率就应该是各个维度的几率的积。积中的每一个因子都是一个1维的高斯分布。



那么原来那个高纬度的高斯分布的covariance matrix \\(\Sigma\\)就变成了一个对角矩阵diagonal matrix 。这样可以减少参数量，得到一个更简单的模型。但是结果不一定好，因为首先要确保各个维度相互独立，不能有关联性。



如果是Binary Feature，可以用伯努利分布。



如果各个维度都是确定是相互独立的，可以用朴素贝叶斯分类器 Naive Bayes Classifier。





#### 分析

![20fenxi](https://i.loli.net/2018/12/13/5c12738626006.png)

将其上下同除以分子。



现在假设 


$$
z=ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}
$$


那么，原式就变成了


$$
\frac{1}{1+e^{-z}}
$$


记


$$
\frac{1}{1+e^{-z}}=\sigma(z) \qquad  ,即\mathrm{Sigmoid\;function}
$$

---

#### 针对\\(z\\)，对z进行一系列化简、转化

![21z](https://i.loli.net/2018/12/13/5c12738630c1b.png)

![22z](https://i.loli.net/2018/12/13/5c12741602ec0.png)

![23z](https://i.loli.net/2018/12/13/5c1274160bb4f.png)

得到最后的结果：蓝色方框里面的\\(z\\)

---



因为 共用了covariance matrix，可以再次简化

![24zz](https://i.loli.net/2018/12/13/5c12740e8245a.png)

![25zz](https://i.loli.net/2018/12/13/5c127415bdbd3.png)

左边那项是一个矩阵(vector)，右边是一项数字(scalar)。



这也解释了为什么共用一个covariance matrix 以后的boundary 是一条直线。

![26reference](https://i.loli.net/2018/12/13/5c127416c8d4b.png)

