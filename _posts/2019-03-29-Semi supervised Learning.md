---
tag: 机器学习
---



* 在supervised learning中，有一大堆的training data，假设有R笔training data(上标记为r)，每一个data都包括了function的input，image；以及output，class labels。

**supervised learning data:**


$$
\{(x^r,\hat{y}^r)\}^R_{r=1}
$$



* 在semi-supervised learning中，除了含有labeled data以外，另外含有一组unlabeled data(上标记为u)。这些data只有function的input而没有output。

**semi-supervised learning extra data**

$$
\{ x^u \}^{R+U}_{u=R}
$$


通常来说，U远大于R，即unlabeled data的数量远大于label data。


**semi-supervised learning通常可以分为两种**

1. Transductive learning：unlabeled data is the testing set. 这里不能算是cheating,因为没有用testing set的label(本身就不含)。只用了testing data的feature。
2. Inductive learning: unlabeled data is not the testing data. 在某些情况下，例如手头没有或者不能动testing data。即在training的时候还不知道testing data会长什么样。所使用的unlabeled data不属于testing data.


使用transductive或者Inductive 就要depend on现在testing data是否已经获得且可用。



**做semi-supervised learning的原因**

通常情况下，并不会缺少data，而是缺少labeled data。如果能够利用好unlabeled data会是很有价值的一件事情。

生物学上的理由（类比人）。



![a1p3](https://wx4.sinaimg.cn/large/8f3e11fcly1g1n6u1h6yjj20nl0hhabl.jpg)







## Semi-Supervised learning works的原因





> e.g. 假设要做一个classification，有两个class



现在有很多(占绝大多数)unlabeled data。不知道标签是猫还是狗。



![b1p4](https://ws4.sinaimg.cn/large/8f3e11fcly1g1n6ucnuuwj20nm0hcdhw.jpg)





如图所示，有颜色的点代表labeled data，现在如果只用labeled data，那么很直觉地就会画一条竖线。



但是如果能够用上unlabeled data，就可能会影响决定。如果unlabeled data的分布如图，那么就可能暗示这个boundary是一条斜线。





显然，semi-supervised learning在使用的时候往往伴随着一些假设，其Performance取决于假设的正确性、合理性。





## Semi-supervised learning for Generative Model





### review



在Generative Model里面，通过朴素贝叶斯公式来求得概率。而在这之前需要知道两个数据。



- \\(P(C_i)\\)



这个根据样本个数比例进行一个估测。







- $$
  P(x|C_i)
  $$

  



后者根据高斯分布来估测。



高斯分布需要知道\\(\mu\\)和
$$
\Sigma
$$






然后根据公式input x就能够求得分别属于某一类的概率。也就可以画出boundary。



![b2p7](https://ws2.sinaimg.cn/large/8f3e11fcly1g1n6ukmt99j20nm0hjgns.jpg)





### 加上Unlabeled data



如果加上如图绿点所示的unlabeled data。



![b3p8](https://ws3.sinaimg.cn/large/8f3e11fcly1g1n6upu7tmj20nf0hhju2.jpg)





此时原来的\\(\mu\\)和
$$
\Sigma
$$
已经是不合理的。





Unlabeled data \\(x^u\\) 会影响
$$
P(C_1)
$$
，
$$
P(C_2)
$$
（data 的数目看起来显然也会有影响，例如图中右边的class的数目就会变多），
$$
\mu^1
$$
，
$$
\mu^2
$$
，
$$
\Sigma
$$
。





这样就会影响posterior probability的式子从而影响Decision Boundary。



### Formulation



#### 首先：根据labeled data初始化一组参数(Gaussion Distribution)



记为
$$
\theta
$$

$$
\theta=\{P(C_1),P(C_2),\mu^1,\mu^2,\Sigma\}
$$




#### Step1：根据已有公式计算每一组unlabeled data的posterior probability


$$
P_{\theta}(C_1|x^{\mu})
$$


#### Step2：更新model

$$
P(C_1)=\frac{N_1+\Sigma_{x^u}P(C_1|x^u)}{N}\\
u^1=\frac{1}{N_1}\sum_{x^r\in C_1}x^r+\frac{1}{\Sigma_{x^u}P(C_1|x^u)}\sum_{x^u}P(C_1|x^u)x^u......
$$



### 背后的理论

#### 假设只有labeled data

要做的事情就是去maximize likelihood，或者去maximize log likelihood。


每一笔single data的likelihood都可以算出来


$$
\log L(\theta)=\sum_{x^r} \log P_{\theta}(x^r,\hat{y}^r)=\sum_{x^r} \log P_{\theta}(x^r|\hat{y}^r)P(\hat{y}^r)
$$





#### 加上unlabeled data



这个likelihood 就会发生变化


$$
\log L(\theta)=\sum_{x^r,\hat{y}^r} \log P_{\theta}(x^r|\hat{y}^r)+\sum_{x^u} \log P_{\theta}(x^u)
$$

其中

$$
P_{\theta}(x^u)=P_{\theta}(x^u|C_1)P(C_1)+P_{\theta}(x^u|C_2)P(C_2)
$$


因为unlabeled data并不知道来自于哪一个class，所以两个都要加上，这才是这笔unlabeled data出现的几率。


但是这个likelihood function并不是convex的，所以需要去iteratively地去solve 他。在每次做完step1和step2以后，就可以让这个likelihood增加一点，到最后就可以收敛在一个local minima。



## Low-density Separation 假设



![c1p11](https://wx3.sinaimg.cn/large/8f3e11fcly1g1n6uw272oj20nn0gijta.jpg)

这个假设认为这个世界"非黑即白"。意思就是说：假设现在有很多data，包含label和unlabeled的data，但是在两个class之间会有一个非常明显的鸿沟，会有一个非常明显的boundary。

如图所示，如果现在只考虑labeled data，蓝色红色点，那么两个boundary都可以，但是加上unlabeled data以后(绿色点)，如果基于这个假设，右边的这个boundary就变得不好了。


顾名思义，Low-density的意思是：在这两个class的交界处他的density是低的。




## Self-training


Low density最具代表性的方法是Self-training



![c2p12](https://ws3.sinaimg.cn/large/8f3e11fcly1g1n6v1i0muj20nr0h90v1.jpg)




如果有一些labeled data和一些unlabeled data，做法是：


1. 先从labeled data去train一个function。可以通过各种方法各种model。
2. 将这个function(model)应用到unlabeled data中，得到output，也就是一组Pseudo-label。让这些unlabeled data具有label。
3. 通过自己设定的规则来选取一些具有Pseudo-label的unlabeled data，并将其直接加到labeled data里面去。也可以给这些data加一些weight。
4. 使用新的labeled data进行训练。


**注：Self-training这一招在Regression上面无效**







## 比较Self-training和generative model里的半监督学习



Self-training得到的label是hard label，而semi-supervised generative model里的是soft label。


**假设在一个neural network中，这里强调神经网络**


![c3p13](https://ws4.sinaimg.cn/large/8f3e11fcly1g1n6v6xjoyj20nh0hewgc.jpg)


通过labeled data训练出一组参数
$$
\theta^*
$$
。



当有一个unlabeled Input 
$$
x^u
$$
经过这一组参数，得到一个output
$$
\begin{bmatrix}
0.7 \\
0.3
\end{bmatrix}
$$
。





* Hard label，因为Class1的几率比较大，干脆直接label成class1，这样会得到新的target (output) 


$$
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$



* Soft label，70%几率Class1，30%几率Class2。这样会得到新的target



$$
\begin{bmatrix}
0.7 \\
0.3 
\end{bmatrix}
$$


在做neural network的时候，下面的方法（使用Soft label）不会有用。因为输出还是和原来一样。






## Entropy-based Regularization


和上面一样，假设有一个unlabeled input 
$$
x^u
$$
经过
$$
\theta^*
$$
得到一个label 
$$
y^u
$$
。
这个label是一个distribution（概率分布）。



根据Low-density的假设，这个output的distribution越集中越好。


![c4p14](https://wx3.sinaimg.cn/large/8f3e11fcly1g1n6vda2odj20no0hhac6.jpg)




如上图左下所示，前两个的分布会比较好。



现在要用一个数值的方法来evaluate这个distribution：



增加一个正则项：Entropy of 
$$
y^u
$$
，来评价这个output分布的集中程度。




其中，在这个例子中，


$$
E(y^u)=-\sum_{m=1}^5 y^u_m \ln(y^u_m)
$$




显然，这个正则项越小越好。


如图所示，前两个的正则项都是0，表示其分布比较好。



**现在可以修改Loss function，假设原来的loss function是cross entropy，现在加上该Entropy-based Regularization**




$$
L=\sum_{x^r}C(y^r,\hat{y}^r) + \lambda \sum_{x^u} E(y^u)
$$



\\(\lambda\\)表示在训练的时候是偏向labeled data多一点还是偏向unlabeled data多一点。







## Outlook:Semi-supervised SVM


![c5p15](https://ws4.sinaimg.cn/large/8f3e11fcly1g1n6vilhghj20nn0hkdhr.jpg)




## Smoothness Assumption



"similar" x has the same label y


![c6p18](https://ws4.sinaimg.cn/large/8f3e11fcly1g1n6vogzrsj20ns0heq60.jpg)



假设：

* x的分布是不平均的，在某些地方很集中，某些地方很分散。
* 如果\\(x^1\\)和\\(x^2\\)在某个high density region很近，那么\\(\hat{y}^1\\)和\\(\hat{y}^2\\)也是很接近的。
* 这里的high density region指的是connected by a high density path，也就是两个x之间有高密度的路径相连接。


如上图右，假设有3笔data，\\(x^1\\)和\\(x^2\\)之间是一个high density path相连的，更可能有相似的label。但是虽然\\(x^2\\)和\\(x^3\\)比较接近，他们之间没有高密度的路径相连接，所以\\(\hat{y}^2\\)和\\(\hat{y}^3\\)不类似。




> 具体的例子



![c7p19](https://ws4.sinaimg.cn/large/8f3e11fcly1g1n6vus9jaj20nr0h5dhx.jpg)



如上图上，左边两个都是2，右边是3，数据是unlabeled data。如果单从像素相似度的角度来看，中间的和右边的是比较相近的，但是这样显然不对。如果考虑其他data连续的变化(high density path)相连接的(有overlap的data一路propagate过去)，那么左边的可以通过连续的小变化变成中间的。


某些data很像但不一定是同一个class，某些data很不像也有可能是同一个class。


## Cluster and then Label


![c8p22](https://wx2.sinaimg.cn/large/8f3e11fcly1g1n6w3pl6nj20no0gtq5e.jpg)

某些简单的情况下，cluster很强的话，可以画出分布，然后圈圈，分别添上label，然后拿来learn。


## Graph-based Approach





**引入Graph-structure，用graph-structure来表达"connected by a high density path" 这件事**


![c9p23](https://wx2.sinaimg.cn/large/8f3e11fcly1g1n6w95lt0j20nq0h2gns.jpg)


* 把所有的data points建成一个graph，每一笔data point x都是图上的点
* 要知道Similarity就要把点之间的edge建出来。




有的时候图的表现是很自然的，

> e.g.

* 通过Hyperlink连接的网页
* 论文的引用关系



很多时候需要自己想办法建这个graph，graph的好坏对结果的影响是非常criticcal的。


图的建法要靠经验和直觉，通常的做法是：

* 定义两个objects之间如何来算相似度，影像如果based on pixels来算相似度performance不会太好，但是可以用auto encoder抽出来的feature来算相似度
* 算完相似度以后可以建graph了，建法有很多种：
    * K Nearest Neighbor 
    * e-Neighborhood
* Edge并不是只有相连和不相连这样的binary的选择，可以给edge一些weight，让edge与连接起来的两个data points之间的相似度成正比。


相似度的定义可以用**Gaussian Radial Basis Function**:

$$
s(x^i,x^j)=\exp(-\gamma \lVert x^i-x^j \rVert^2)
$$


在经验上取exponential会有比较好的performance。因为这个function下降的速度是非常快的，所以只有当\\(x^i\\)和\\(x^j\\)非常靠近的时候similarity才会大，距离稍微一远就会下降得很快变得很小。这样可以避免跨海沟(距离较远)的link。


![c10p24](https://ws1.sinaimg.cn/large/8f3e11fcly1g1n6wdnwkoj20gl09ot99.jpg)




### 更进一步

![c11p25](https://ws1.sinaimg.cn/large/8f3e11fcly1g1n6whls4oj20ni0hamzk.jpg)


如上图，如果在某个graph上面有一些label data，假设是属于class1,那么跟他相连的这些data points是class1的几率也会上升。

但是光影响到邻居是不够的(帮助不会太大)，因为相连的东西本来output就会很像。


graph-based 真正有很大帮助的是：Class是可以传递的，这种"相似"会透过graph link传递(propagate)过去。如上图右上，某个蓝色的点会通过连接影响到和他有通路的data points。


graph-based也要求收集到的数据要够多，否则的话就有可能传不过去。


### 定量使用graph


**在graph structure上面定义一个labels的smoothness**


![c12p26](https://wx4.sinaimg.cn/large/8f3e11fcly1g1n6wls86jj20nj0hd767.jpg)


如上图下，左边的显然比右边的更加smooth。


定义


$$
S=\frac{1}{2}\sum_{i,j} w_{i,j}(y^i-y^j)^2
$$



将所有有连接的data points全部加起来，\\(w_{i,j}\\)是两个data points之间的weight。\\(\frac{1}{2}\\)是为了之后计算方便。


这个值越小越smooth。


根据这个式子计算出来的结果符合预期，左边的比右边的小。



**这个式子可以稍微整理一下，变成一个比较简洁的式子**

把y(包括labeled data和unlabeled data,R+U个dimensions)，串成一个vector。

$$
y=\lbrack \cdots y^i \cdots y^j \cdots \rbrack^T
$$


所以上面的计算smoothness的式子可以写成


$$
S=\frac{1}{2}\sum_{i,j} w_{i,j}(y^i-y^j)^2=y^TLy
$$



其中，L是一个(R+U)乘(R+U)的matrix，称为Graph Laplacian。




$$
L=D-W
$$



例如上图右下，其中，




$$
W=\begin{bmatrix}
0&2&3&0 \\
2&0&1&0 \\
3&1&0&1 \\
0&0&1&0\\
\end{bmatrix}
$$



W就是图的矩阵表示（邻接矩阵）。





其中，


$$
D=\begin{bmatrix}
5&0&0&0 \\
0&3&0&0 \\
0&0&5&0 \\
0&0&0&1 \\
\end{bmatrix}
$$


D就是每一个结点的出度，将每一行加起来放在diagonal的地方，组成一个diagonal的matrix。



将这个简洁的式子左边展开以后就得到了右边的这个式子。



**可以用这个简洁的式子来evaluate 得到的label有多smooth。**




$$
S=\frac{1}{2}\sum_{i,j} w_{i,j}(y^i-y^j)^2=y^TLy
$$


其中的这个\\(y\\)和\\(y^T\\)是depend on network parameters的，所以可以将这个项作为一个regularization term，用来优化参数。


在原来的loss function里面添加上一项。

$$
L=\sum_{x^r}C(y^r,\hat{y}^r) +\lambda S
$$



这一项包括了所有labeled data和unlabele data的smoothness值。



![c13p28](https://wx1.sinaimg.cn/large/8f3e11fcly1g1n6wqfydnj20nm0hdtax.jpg)



如上图右下，算smoothness的这一步不一定要放在output的地方，可以放在network的任何地方。






