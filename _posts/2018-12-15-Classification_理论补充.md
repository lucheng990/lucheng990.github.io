---
tag: 机器学习
---



# Logistic Regression

---

## 理论补充

对于这个Posterior Probability
$$
P(C_1|x)
$$

$$
P(C_1|x)=\frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}
$$


可以发现，当分子分母同除以分子以后，


$$
P(C_1|x)=\frac{1}{1+\frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}}
$$


记


$$
z=\ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}
$$


那么，原式


$$
P(C_1|x)=\frac{1}{1+e^{-z}}
$$


记


$$
\frac{1}{1+e^{-z}}=\sigma(z) \qquad  即：\mathrm{Sigmoid\; function}
$$


> 在信息科学中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间。



对于z来说


$$
z=\ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}=\ln\frac{P(x|C_1)}{P(x|C_2)}+\ln\frac{P(C_1)}{P(C_2)}
$$


其中


$$
\ln\frac{P(C_1)}{P(C_2)}=\frac{\frac{N_1}{N_1+N_2}}{\frac{N_2}{N_1+N_2}}=\frac{N_1}{N_2}
$$


而


$$
P(x|C_1)=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma^1|^{1/2}}exp{\{-\frac{1}{2}(x-\mu^1)^T(\Sigma^1)^{-1}(x-\mu^1)\}}
$$

$$
P(x|C_2)=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma^2|^{1/2}}exp{\{-\frac{1}{2}(x-\mu^2)^T(\Sigma^2)^{-1}(x-\mu^2)\}}
$$


其中，D表示维度，竖线表示模，上标-1表示 逆矩阵，T表示矩阵的转置。



相除，再取自然对数：


$$
\ln\frac{\frac{1}{|\Sigma^1|^{1/2}}exp{\{-\frac{1}{2}(x-\mu^1)^T(\Sigma^1)^{-1}(x-\mu^1)\}}}{\frac{1}{|\Sigma^2|^{1/2}}exp{\{-\frac{1}{2}(x-\mu^2)^T(\Sigma^2)^{-1}(x-\mu^2)\}}}
$$


进行化简，并把exp两项合并并提出，得到：


$$
\ln\frac{|\Sigma^2|^{1/2}}{|\Sigma^1|^{1/2}}-\frac{1}{2}[(x-\mu^1)^T(\Sigma^1)^{-1}(x-\mu^1)-(x-\mu^2)^T(\Sigma^2)^{-1}(x-\mu^2)]
$$


对于后面的一项，


$$
(x-\mu^1 )^T(\Sigma^1)^{-1}(x-\mu^1)\\
\begin{align*}
&=x^T(\Sigma^1)^{-1}x-x^T(\Sigma^1)^{-1}\mu^1-(\mu^1)^T(\Sigma^1)^{-1}x+(\mu^1)^T(\Sigma^1)^{-1}\mu^1\\
&=x^T(\Sigma^1)^{-1}x-2(\mu^1)^T(\Sigma^1)^{-1}x+(\mu^1)^T(\Sigma^1)^{-1}\mu^1
\end{align*}
$$


同理


$$
(x-\mu^2 )^T(\Sigma^2)^{-1}(x-\mu^2)\\
\begin{align*}
&=x^T(\Sigma^2)^{-1}x-x^T(\Sigma^2)^{-1}\mu^2-(\mu^2)^T(\Sigma^2)^{-1}x+(\mu^2)^T(\Sigma^2)^{-1}\mu^2\\
&=x^T(\Sigma^2)^{-1}x-2(\mu^2)^T(\Sigma^2)^{-1}x+(\mu^2)^T(\Sigma^2)^{-1}\mu^2
\end{align*}
$$


所以呢


$$
z=\ln\frac{|\Sigma^2|^{1/2}}{|\Sigma^1|^{1/2}}-\frac{1}{2}x^T(\Sigma^1)^{-1}x+(\mu^1)^T(\Sigma^1)^{-1}x-\frac{1}{2}(\mu^1)^T(\Sigma^1)^{-1}\mu^1\\
+\frac{1}{2}x^T(\Sigma^2)^{-1}x-(\mu^2)^T(\Sigma^2)^{-1}x+\frac{1}{2}(\mu^2)^T(\Sigma^2)^{-1}\mu^2+\ln\frac{N_1}{N_2}
$$


看起来非常复杂，但是一般来说covariance matrix 可以共用，所以


$$
\Sigma^1=\Sigma^2=\Sigma
$$


![01](https://i.loli.net/2018/12/15/5c150de722f9c.png)



剩下的项还可以进行合并，所以可以得到：


$$
z=(\mu^1-\mu^2)^T\Sigma^{-1}x-\frac{1}{2}(\mu^1)^T\Sigma^{-1}\mu^1+\frac{1}{2}(\mu^2)^T\Sigma^{-1}\mu^2+\ln\frac{N_1}{N_2}
$$


记

![02](https://i.loli.net/2018/12/15/5c150de714aec.png)

所以x的系数项是一个vector，把这个vector 记为 
$$
w^T
$$
，而后面的那一长串记为
$$
b
$$
，就是一个数字scalar



所以,posterior probability就成了：


$$
P(C_1|x)=\sigma(w\cdot x+b)
$$


这里基于一个前提：两者的covariance matrix相同，这也解释了covariance matrix相同的时候的boundary是一条直线。



所以，在 generative model 里面，只要能够估计出 
$$
N_1,N_2,\mu^1,\mu^2,\Sigma
$$
，就可以得到
$$
w^T\;和\;b
$$
，就可以算几率。



虽然有更好的方法..........
