---
tag: 机器学习
---



# Gradient Descent 

## Review 

Gradient Descent 算法：

![01Review](https://i.loli.net/2018/12/04/5c06963d42744.png)

![02visualize](https://i.loli.net/2018/12/04/5c06965c7f6c8.png)



## Gradient Descent 中的技巧

### Tuning your learning rates

 1. learning-rate 太小的话，参数更新太慢
 2. learning-rate 太大的话，更新步伐太大，有可能会卡在某一个点，无法到最低点，
 3. learning-rate 太太大，会飞出去
 4. 在做梯度下降的时候，如果能够画出loss函数对更新轮数或者loss函数的变化对更新轮数的图，有利于调整learning-rate。

![03somecases](https://i.loli.net/2018/12/04/5c0696731e942.png)





### 解决方法：Adaptive learning-rate（自动调节learning-rate)

在刚开始的时候，比较希望有一个较大的learning-rate ，随着训练的进行，希望Leaning-rate 不断减少。

> 最简单的方法：增加一个时间dependent 函数：e.g.

$$
\eta^t=\frac{\eta}{\sqrt{t+1}}
$$

![04Adaptive](https://i.loli.net/2018/12/04/5c069689395b7.png)



> Vanilla Gradient Descent 

$$
w^{t+1}\gets w^t-\eta^tg^t   \\
其中 \\
\eta^t=\frac{\eta(学习率)}{\sqrt{t+1}}  \qquad,\\
g^t=\frac{\partial{L(\theta^t)}}{\partial{w}}
$$

Vanilla Gradient 没有对每个参数单独设置learning-rate



> Adagrad：每个参数都有不同的learning-rate

$$
w^{t+1}\gets w^t-\frac{\eta^t}{\sigma^t}g^t\\
其中\\
\eta^t=\frac{\eta（学习率）}{\sqrt{t+1}}   \qquad,\\
g^t=\frac{\partial{L(\theta^t)}}{\partial{w}} \qquad,\\
\sigma \;是过去所有的对应该参数的偏微分值的\mathrm{root\,mean\,square}
$$

![05Adagrad](https://i.loli.net/2018/12/04/5c0696a2c5df2.png)

例如：

![06Adacases](https://i.loli.net/2018/12/04/5c06989ca49d5.png)

其中，\\(\eta^t\\)是一个时间dependent 的参数，\\(\sigma^n\\)是一个参数dependent的参数(与之前的所有参数的偏微分有关)。



回到公式，里面的\\(\sqrt{t+1}\\)可以约分

![07yuefen](https://i.loli.net/2018/12/04/5c0696eba97ba.png)

所以可以简化公式:
$$
w^{t+1}\gets w^t-\frac{\eta}{\sqrt{\Sigma^t_{i=0}(g^i)^2}}\,g^t
$$


**矛盾**： \\(g^t\\) 表明梯度越大更新越快，而分母表明梯度越大更新越慢，两者形成反差。

![08contradiction](https://i.loli.net/2018/12/04/5c0698ba46fe4.png)

> 其中一种 Intuitive Reason： 反差

![09fancha](https://i.loli.net/2018/12/04/5c0698cfa966d.png)

分母项把造成的反差给抵消了。





更加正式的解释：假设是一个二次函数模型，从某一点\\(x_0\\)到最低点，水平距离为\\(x_0+\frac{b}{2a}\\\)。恰巧就是一次微分除以二次微分。可以证明一次微分除以二次微分同阶近似于 Adagrad的更新幅度。

![10ercihanshu](https://i.loli.net/2018/12/04/5c069a189c3e2.png)





更为直观地，考虑两个参数

![11ercivisualize](https://i.loli.net/2018/12/04/5c069a805f7ab.png)

\\(w_1\\)上a点的微分值较小，\\(w_2\\)上c点的微分值较大，但是a点比c点更加远离最低点，所以单单考虑一次微分的值不显得很好。



再考虑二次微分，a处一次微分小但是二次微分也小，相比而言c处一次微分大但是二次微分也大。做除法以后将更为合理。

![12beststep](https://i.loli.net/2018/12/04/5c069a31abefd.png)

最好的步伐要考虑到二次微分。



而Adagrad 里面的分子项\\(\sqrt{\Sigma^t_{i=0}(g^i)^2}\\)其实可以近似代替二次微分。优点：使用以前算过的数据，减小计算量。



**用一阶导数和的平方根来估计二阶导数只是一种近似，这样的考虑基于两者成正比关系**

![13firstandsecond](https://i.loli.net/2018/12/04/5c069a4d2f7e6.png)

可以看到，在一次微分上面取很多的sample，如果它的平方和开根号大，那么相应的二次微分也大。所以可以用来近似估计二次微分大小。





### Stochastic Gradient Descent （随机梯度下降）

> Stochastic training  可以让训练更快速

![14Stochastic](https://i.loli.net/2018/12/04/5c069a4d4d043.png)

修改Loss Function ：每次只取一个example,(随机或者按照顺序)，每轮训练只考虑一个example。看一个example就更新一次参数。



### Feature Scaling

假设 Regression 的 Function 里面有两个 Feature ： \\(x_1\\)和\\(x_2\\)。如果两者分布的range很不一样，可以做Scaling，把range分布变成一样。

![15FeatureScaling](https://i.loli.net/2018/12/04/5c069a5ec1e07.png)

> e.g.

![16Featuree.g.](https://i.loli.net/2018/12/04/5c069a4e28098.png)

如图，假设\\(x_1\\)的输入range比较小，为1,2,.......；\\(x_2\\)的range比较大，为100,200,.......



画出Loss_surface,会发现,	如果对\\(w_1\\)和\\(w_2\\)做同样的更动，\\(w_1\\)的变化对y的变化而言比较小，而\\(w_2\\)对y的变化比较大。如左下图，\\(w_1\\)对y的影响（对Loss的影响）比较小，所以在\\(w_1\\)的方向上比较平滑，而在\\(w_2\\)的方向上比较陡峭。这样会有如下缺点：



1. 不用Adagrad 很难搞定,update参数比较难

2. 并不一定是向着最低点走。而是走等高线的方向。



如果对某一个Feature进行缩放（例如\\(w_1\\))，会变成一个圆形（如右下图），这样每次更新都可以向着圆心（最低点）走。





### Feature Scaling 方法

![17FeatureScalingMethod](https://i.loli.net/2018/12/04/5c069a4ad1a22.png)

1. 对全部样本的某一个维度（特征），取其平均：\\(m_i\\)
2. 计算全部样本的该维度下的 标准差(Standard deviation \\(\sigma_i\\))
3. 更新


$$
x^r_i\gets \frac{x^r_i-m_i}{\sigma_i}
$$




### Gradient Descent Theory

![18Theory](https://i.loli.net/2018/12/04/5c069a8219af2.png)

目标：在图中找最低点。有一种方法：

1. 在起始点附近画一个红色圈圈，并在圈圈内找一个最低点（比如说在边上）。然后更新参数。

2. 以这个最低点为圆心，再画一个圈圈，并找出最低点。再更新参数。

3. 重复1和2



**现在的问题是：怎么很快地在红色的圈圈内找一个可以让Loss 最小的参数。**



> Taylor Series

![19Taylor](https://i.loli.net/2018/12/04/5c069a2d6fd48.png)

用泰勒展开，可以近似的估计在某个圆心周围的函数



> Multivariable Taylor Series :多参数的泰勒公式



![20MultiTaylor](https://i.loli.net/2018/12/04/5c069b65b20a3.png)



**回到前面的图**，如果红色圈圈足够小

![21BackToFormal](https://i.loli.net/2018/12/04/5c069b9c23519.png)



可以用泰勒公式简化 Loss Function。

![22](https://i.loli.net/2018/12/04/5c069b9e5b0ea.png)

![23](https://i.loli.net/2018/12/04/5c069b8609e16.png)

要在红色圈圈内找到Loss 最小值。

1. 基本是在边界上

2. 将其化成向量形式，\\((\Delta\theta_1,\Delta\theta_2)和 (u,v)\\)。

3. 变成求两个向量的积最小。

4. 选\\((\Delta\theta_1,\Delta\theta_2)\\) 的方向与 \\(u,v\\)相反，且向量的长度（模）为半径（**长度的调节可以靠Learning-rate来调节**）。





**这个式子就是Gradient Descent**

![24](https://i.loli.net/2018/12/04/5c069b8b07605.png)

前提就是这个式子中的 Taylor Series要成立，即红色圈圈足够小。而Learning-rate与红色圈圈的半径成正比，所以Learning-rate 不能太大。

**若果把Taylor Series 里面的二次式考虑进来，可以调大Learning-rate**，但是实际上会增加很多运算。



### Gradient Descent 的限制

![25xianzhi](https://i.loli.net/2018/12/04/5c069b9e447d5.png)

