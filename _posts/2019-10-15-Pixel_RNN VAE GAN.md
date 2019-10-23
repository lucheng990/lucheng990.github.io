---
tag: 机器学习
---











# Generation Model





Generation Model主要有下面三种方法。



* PixelRNN
* Variational Autoencoder(VAE)
* Generative Adversarial Network(GAN)







## Pixel RNN





PixelRNN用来绘画图片，每次生成一个pixel。





### 例如，现在要让machine生成一张3(pixels)*3(px)的图片。



1. 先随机给它一个像素（RGB像素，一个三维的向量）
2. 训练一个model，input为第一个像素，output为第二个像素
3. 再次用2中的model，input为前两个像素，output为第三个像素
4. 不断重复，每次生成一个pixel。直到生成9个pixels。







**利用RNN来处理上面2中的variable length的input**







训练这种model的时候不需要任何label，完全是一个unsupervised任务。只要收集一大堆image就行了。



在实际做法中，如果单纯用一个三维向量来表示每一个像素的颜色，最后生成的图片会比较灰（RGB三个数值差不多）。更好的做法是，用一个1-of-N encoding来表示这些颜色（需要先做Clustering，把几个RGB像素相似的颜色cluster在一起）。





在应用模型的时候，可以给一张遮住的图片进行测试，也可以在开头加一些random的pixels。



*这个方法也可以被用在语音合成上面（WaveNet）*







## VAE









在regularized autoencoders中，高纬数据经过NN上的bottleneck被压缩并提取出关键信息。如果直接拿它作为*generative model*的话，在generative model的前提下，有一个问题，就是**整个网络输入的不连续造成了bottleneck code的latent space是不连续(continuous)的，所以生成模型中很难知道两个code点之间的会生成什么东西**



Variational autoencoder建立在autoencoder之上，如下图所示，decoder部分保持不变，但是encoder部分的输出变成了两个vector，分别是\\(\mu\\)和\\(\sigma\\)。这两个向量的大小（维度）分别都是原bottleneck的大小。







![a1](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic/21GAN/a1.png)



假设现在的bottleneck是三维的，VAE的encoder部分就会输出两个向量，分别是：




$$
\begin{bmatrix}
m_1 \\
m_2 \\
m_3 
\end{bmatrix}
\qquad \mathrm{和}  \qquad
\begin{bmatrix}
\sigma_1 \\
\sigma_2 \\
\sigma_3 
\end{bmatrix}
$$



其中，这里的\\(\mu\\)代表原来encoder的code,\\(\sigma\\)代表了noise的variance，后面还需要做exponential。





然后从一个normal destribution中sample出来一些点\\(e\\)：

$$
\begin{bmatrix}
e_1 \\
e_2 \\
e_3 
\end{bmatrix}
$$




将\\(\sigma\\)取exponential，乘上\\(e\\)，得到的结果加上\\(\mu\\)就得到了新的bottleneck code。*这就相当于在code上面加上了noise，加上了noise相当于是input了一片区域而非某些离散的点，这以后还要能够reconstruct回原来的图像*。这样就达到了可以让输入的bottleneck code连续的目的。




$$
c_i = \exp{\sigma_i} \times e_i +m_i  =  \begin{bmatrix}
c_1 \\
c_2 \\
c_3 
\end{bmatrix}
$$



**其中，\\(\mu\\)和\\(\sigma\\)中的第i个element是bottleneck中的第i个维度的mean和standard deviation，\\(c\\)是加上noise以后的code。**









**这里有一个非常关键的trick，在minimize reconstruction error的同时，要minimize下面这项constraint**

$$
\sum^3_{i=1} (\exp(\sigma_i)-(1+\sigma_i)+(m_i)^2)
$$



因为，如果直接去triain的话，machine会让variance趋于0，这样就和原来的autoencoder没差，所以需要在loss function中加上一些constraint，来强迫variance不可以太小。







![a2](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic/21GAN/a2.png)





如上图所示，绿色的线是\\(e^{\sigma_i}\\)，红色的线是
$$
(1+\sigma_i)
$$
，两者一减得到绿色的线，很明显当\\(\sigma\\)等于0的时候，可以达到minima。因为在得到noised code的时候要对\\(\sigma\\)做exponential，所以最后的variance是1。最后的\\(m\\)是regularization term。



---



## 进一步的解释









### Gaussion Mixture Model









现在假设有一个distribution如下图，黑色曲线所示的复杂的distribution可以看成是很多个Gaussion按照不同的weight组合起来的结果，只要Gaussion的数量足够多，就可以组合成很复杂的distribution。







![a3](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic/21GAN/a3.png)





如上图所示，黑色的distribution可以写成formulation:




$$
P(x) = \sum_m P(m)P(x|m)
$$




上式表明，要从这个Gaussion mixture model里面sample出某一个data，步骤是：



1. 从一个multinomial的distribution中去决定要从哪一个部分的gaussion中去sample，假设是第m个。


$$
m \sim P(m) 
$$


2. 从选定的Gaussion distribution中sample出data points。




$$
x|m \sim N(\mu^m,\Sigma^m)
$$






3. Summation over所有的可能m的取值（所有的gaussion），即得到最开始的式子：




$$
P(x)  = \sum_m P(m)P(x|m)
$$








在Gaussion Mixture Model里面，需要决定Mixture的数目





Gaussion mixture model这件事更像是一件classification的事情，每一个x都来自于某一个分类（gaussion model）。





在分类这件事情上，单做clustering是不够的，使用**distributed representation**可以取得更好的结果（从无数多个Gaussion中去sample data）。







___







VAE的做法是：



1. 从一个Normal distribution中sample出一个vector \\(z\\)。其中，\\(z\\)的不同维度表示了各种不同的attribute。




$$
z \sim N(0,I)
$$






2. 根据\\(z\\)来决定另外一个normal distribution。




$$
x|z \sim N(\mu(z), \sigma(z))
$$






要从\\(z\\)得到这个新的distribution，可以用一个NN来表示，这个NN的input是\\(z\\)，output是新的Gaussion distribution的mean和variance，这里可以把variance拉直或者取diagonal的地方的值拉直。因为\\(z\\)是continuous的，有无穷多个取值，所以这个新的normal distribution也有无穷多种取值，这种做法就实现了类似无穷多个Gaussion叠加的效果。









3. 得到最后的几率分布








$$
P(x) = \int\limits_{z} P(z)P(x|z)dz
$$






上面的从\\(z\\)得到一个gaussion distribution的过程是decoder的部分



![a4](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic/21GAN/a4.png)









### Encoder部分





引入另外一个distribution，




$$
q(z|x) \sim N(\mu'(x),\sigma'(x))
$$




![a5](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic/21GAN/a5.png)





如上图所示，encoder部分的NN的input是x，output是另外一个normal distribution。









## Find out the best model



要训练整个encoder和decoder model，可以使用Maximizing Likelihood的方法，其中，loss function可以是几率:




$$
L = \sum_x \log{P(x)} \tag{0}
$$












在VAE中，要得到某一个\\(x\\)的几率可以写成：






$$
P(x) = \int_{-\infty}^{+\infty} P(z) P(x|z) dz   \tag{1}
$$






同时，对于\\(P(x)\\)本身，也可以做一些变形：






$$
\begin{align*}

&\log{P(x)} = \log{P(x)} \\
\Longrightarrow &  \log{P(x)} = \int\limits_z q(z|x) \log{P(x)}dz

\end{align*}
$$





上式成立的原因是，首先在讨论P(x)的时候，是属于模型的评价（应用）阶段，此时x是确定的，所以\\(P(x)\\)是一个常数；而
$$
q(z|x)
$$
可以是一个任意的概率分布，它的概率密度求积分就是1。将\\(P(x)\\)这个常数放进去。







继续做一些变形：







$$
\begin{align*} 
 \log{P(x)} & = \int\limits_z q(z|x) \log{P(x)}dz \\
 

& = \int\limits_z q(z|x) \log{(\frac{P(z,x)}{P(z|x)})} dz \\
& = \int\limits_z q(z|x)\log{(\frac{P(z,x)}{q(z|x)}\frac{q(z|x)}{P(z|x)})} dz \\
& = \int\limits_z q(z|x)\log{(\frac{P(z,x)}{q(z|x)})}dz + \int\limits_z q(z|x)\log{(\frac{q(z|x)}{P(z|x)})} dz \\
& = \int\limits_z q(z|x)\log{(\frac{P(z,x)}{q(z|x)})}dz + KL(q(z|x) || P(z|x)) \\
& = L_b + KL(q(z|x) || P(z|x))  \tag{2} \\
& \ge  L_b =  \int\limits_z q(z|x)\log{(\frac{P(x|z)P(z)}{q(z|x)})}dz
\end{align*}
$$














如上式，右边的式子是一个**KL divergence**，代表这两个distribution的相近程度，或者说两个distribution之间的距离。（两个越相近的分布的KL值越小，但是恒大于等于0。如果两个distribution一模一样的话，log这一项就等于0）所以最后得到的式子是原式的**L lower bound**，称为\\(L_b\\)。





现在，要maximizing的对象变成了：










$$
\log{P(x)} = L_b + KL(q(z|x)||P(z|x))
$$











其中，






$$
L_b =\int\limits_z q(z|x)\log{(\frac{P(x|z)P(z)}{q(z|x)})}dz
$$



在\\(L_b\\)这个式子中，\\(P(z)\\)是已知的（例如是一个normal distribution，**这里的前提是知道了Input x，由vae的encoder部分来得到关于z的高斯分布的两个参数，这样就知道z的概率密度函数。**），而未知的是
$$
P(x|z)
$$
和
$$
q(z|x)
$$
。





在(1)式中，只需要找
$$
P(x|z)
$$
这一项就够了，但是在这里(2)式中，需要找P和q两项，这样做的原因是：确保在maxmizing lower bound \\(L_b\\)的同时能够maxmizing这个likelihood。







---





#### 对于这一点的解释：





loss value(likelihood)的构成如下图左，由KL和\\(L_b\\)构成：



![a6](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic/21GAN/a6.png)





由(0)和(1)式可知，整个likelihood value与P有关而与q无关。所以q的值不影响整个likelihood。在此基础上，如果**固定**
$$
P(x|z)
$$
，那么此时通过调整q这个NN中的参数来maximizing lower bound的话，\\(L_b\\)的值增长了而整体的likelihood不变，那么lower bound就会不断地接近likelihood。这样就能够在maximize lower bound的同时保证maximize likelihood。如上图右所示。





---









对于这个\\(L_b\\)：






$$
\begin{align*}
L_b & = \int\limits_z q(z|x)\log{(\frac{P(z,x)}{q(z|x)})}dz \\
& = \int\limits_z q(z|x)\log{(\frac{P(x|z)P(z)}{q(z|x)})}dz \\
& = \int\limits_z q(z|x)\log{(\frac{P(z)}{q(z|x)})}dz + \int\limits_z q(z|x)\log{P(x|z)}dz\\
&= -KL(q(z|x)||P(z))+\int\limits_z q(z|x)\log{P(x|z)}dz


\end{align*}
$$






* 最后的左式也是一个KL divergence，如果要minimize这个KL divergence，等价于minimize 下面这项：






$$
\sum_i (\exp(\sigma_i)-(1+\sigma_i)+(m_i)^2)
$$


> Refer to the appendix B of the original VAE paper







* 最后的右式






$$
\begin{align*}

&\int\limits_z q(z|x)\log{P(x|z)}dz \\
= & E_{q(z|x)}[\log{P(x|z)}]

\end{align*}
$$




要maximize这项，就是auto encoder做的事情，







![a7](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic/21GAN/a7.png)





结果需要input和最后的output越接近越好。













---









VAE产生image的动机是要和data base里面的越接近越好，而且在描述相似度的时候会存在一定的不准确性。或者说，VAE只是在模仿原来的image，而并不是在真正产生images。







# Generative Adversarial Network





 

GAN是非监督式学习的一种方法，通过让两个神经网络相互博弈的方式进行学习。其由一个生成网络与一个判别网络组成。生成网络从latent space中随机取样作为输入，其输出结果需要尽量模仿训练集中的真实样本。判别网络的输入则为真实样本或生成网络的输出，其目的是将生成网络的输出从真实样本中尽可能分辨出来（一个二分类的分类模型）。而生成网络则要尽可能地欺骗判别网络。两个网络相互对抗、不断调整参数，最终目的是使判别网络无法判断生成网络的输出结果是否真实。





------





训练GAN模型:







1. 首先有一个第一代的NN Generator，可以是random的。这个NN的架构和VAE中的Decoder一样，input是从一个distribution中sample出来的一些向量。然后通过这个Generator生成一些Images（fake images)。
2. 接下来有一个第一代的NN Discriminator。Fix Generator的参数，分别从Generator生成的图片(fake images)以及real images中采样正负样本作为输入，然后训练这个binary classification的NN。
3. 开始训练Generator，此时从最开始那个distribution中randomly sample一些vector，**fix住Discriminator的参数**，optimize Generator中的参数使得最后Discriminator的output为1（real value）。
4. 重复2和3的这两个过程。









**这里有一个和VAE的很大的区别就是：Generator没有看过data base里面的real images。所以最后Generator可以产生出一些images是data base里面从来没有过的。**







------





下面是一些demo:





> http://cs.stanford.edu/people/karpathy/gan/
>
> https://openai.com/blog/generative-models/














































































