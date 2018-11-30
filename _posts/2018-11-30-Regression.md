---
tag: 机器学习
---





#### Regression:Output a scalar
> e.g.
- 股票预测
- 自动驾驶（output an angle of steering wheel)
- 商品推荐系统
- 进化后属性预测


##### 具体步骤(以神奇宝贝预测进化后cp值为例)
1. Step1:Model（A set of function)  

    假设Linear model:  \\(y=b+w*x_{cp}\\) ——其中w 和b 是参数可以带入各种不同的值，因此function有无穷多个。

    > 线性模型：\\(y=b+\sum w_ix_i \quad (w_i:weight, b_i: bias)\\)





2. Step2:Goodness of Function (衡量Function 的好坏)	

    - 首先准备好 *training data*

    > Training data 含有 \\(x_cp \quad 以及\bar{y}, \bar{y}：标签，即正确的值\\)

    - 然后定义Loss Function *L*

    > *L* :input a function ; output how bad it is ,即：*L*是函数的函数

    可以定义Loss Function:
    $$
    \mathrm{L}(f)=\sum_{n=1}^{10}(\bar{y}^n-f(x_{cp}^n))^2
    =\mathrm{L}(w,b)=\sum_{n=1}^{10}(\bar{y}^n-(b+w*x_{cp}^n))^2
    $$
    ![01Loss_Function](https://i.loli.net/2018/11/30/5c014afc78767.png)

    其中，\\(\bar{y}\\)和\\(x_{cp}^n\\) 的上标n为第n组数据。

    * 若将Loss Function 可视化（visualize），图上每个点代表着一组Loss Function, 越接近X Loss value 越小。

    ![02lossvisualization](https://i.loli.net/2018/11/30/5c014b1854452.png)

3. Step3:Best Function

   * 为了使函数尽量靠近X（loss value 最小的点），可以列出方程
     $$
     f^*=arg\ min\,\mathrm{L}(f)
     $$



   这个函数的意思是，在A set of Function 里面找到一个函数 f*，能最小化损失函数\\(\mathrm{L}(f)\\)。即找到一个最合适的w和b。

   > 求解这个方程可以用到线性代数很快解出（但只适用于 Linear Regression)



   * 更加general的方法： **Gradient Descent**，可以用在各种不同的task上，只要Loss Function对输入的参数 可以进行微分

##### Gradient Descent

![03GD](https://i.loli.net/2018/11/30/5c014b2c02766.png)

假设Loss Function 只有一个参数w ，要使Loss Function 最小，就是要求出他的一个极小值点。

随机选取一个初始值，计算偏微分（导数），如果小于0说明左高右低，所以向右走(增加参数大小)



![04GD](https://i.loli.net/2018/11/30/5c014b3aa2a43.png)

\\(\eta\\)，即学习率(Learning-rate)，用来控制学习的快慢

会存在的问题：只能找到极小值点而不是最小值点。这也与最开始初始值的选取有关。



如果有两个及以上的参数，需要求偏微分，分别更新参数值。可以写成向量形式。

![05GD](https://i.loli.net/2018/11/30/5c014b505cc08.png)

![06GDvisualization](https://i.loli.net/2018/11/30/5c014b64d0ec1.png)



箭头方向即数学上讲的梯度方向。向蓝色方向进。

Gradient Descent 存在的两个问题

* 在导数（偏导数）很小的时候移动非常缓慢.
* 困在极小值点



##### Formulation 

![07GD](https://i.loli.net/2018/11/30/5c014b6ff0b7a.png)

经过学习以后，在testing data 上的loss value 略大于 training data 上的loss value 是合理的。

![08GD](https://i.loli.net/2018/11/30/5c014b815ec86.png)

![09GD](https://i.loli.net/2018/11/30/5c014b8fec999.png)

![10GD](https://i.loli.net/2018/11/30/5c014b9a6798b.png)

![11GD](https://i.loli.net/2018/11/30/5c014ba42d42e.png)

越复杂的Model 可以得到更低的 training data 的loss value，但是不一定会在testing data 上是最好的，容易过拟合。



解决方法

* 收集更多的data
* 设计模型时考虑更多的因素，重新设计Model.



* 考虑到种类因素



![12GD](https://i.loli.net/2018/11/30/5c014bb072827.png)

* 设计冲击函数
* Regularization(**正则化**)





##### Regularization

重新设计Loss函数

![13GD](https://i.loli.net/2018/11/30/5c014bbc64173.png)

这样设计的意图是：不只是让L最小，同时还要让参数w最小（这里没有考虑bias)

w很小意味着它是一个比较平滑的函数，这样的话输入有变化那么输出的变化会很小。



我们相信更加平滑的函数更可能是正确的，λ越大越平滑，但是不能太大。

![14GD](https://i.loli.net/2018/11/30/5c014bc823bc5.png)

λ加大，training data上的loss value 加大是因为同时顾及到了 函数平滑以及loss 最小。
