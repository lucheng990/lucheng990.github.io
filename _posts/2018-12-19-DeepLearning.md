---
tag: 机器学习
---



## Three Steps for Deep Learning

1. **Neural Network**
2. **Goodness of function**
3. **pick the best function**



和Machine-Learning十分类似





### Define a Function



* 通过多个**Logistic Regression** 连接成了**神经网络**，每个Logistic Regression 都有自己的**Weight**和**Bias**，这些所有的**Weight**和**Bias**构成了这个Network的参数\\(\theta\\)。

* 不同的连接方式会有不同的结构





> 最常见的连接方式：Fully Connect Feedforward Network



#### 全连接深度前馈网络





![01FullyConnect](https://i.loli.net/2018/12/19/5c19c42833057.png)



1. 将neuron排成一排一排的（竖向）。
2. 每个neuron都含有一组weight和bias，都是根据training data找出来的。
3. 可以将每个neuron看成是一个Function，input是一个Vector，output也是一个Vector。
4. 给定network structure,其实就是define a function set。接下来就是要进行下面两步操作了。





![02](https://i.loli.net/2018/12/19/5c19c428416e6.png)

将每一排的neuron称为一个Layer



中间的层被称为 Hidden Layers



> Deep=Many hidden Layers



### Matrix Operation

neural network 的运作常常用矩阵运算来表示。



![03matrix](https://i.loli.net/2018/12/19/5c19c428416ed.png)



上图假设每个neuron 里面的activation function都是Sigmoid Function (\\(\sigma\\))。



每层中：常将一个Logistic Regression 中的一组参数作为一行，这一层中有几个神经元那么就有几列。



输入：常将一个Input（含有多个维度）作为一个列向量输入。



bias：把每一层中的bias写成一个列向量



输出：output结构和Input类似，也是一个列向量。





![04matrix](https://i.loli.net/2018/12/19/5c19c4285252f.png)

整个Neural Network的运算就是一连串的Matrix Operation的运算。可以用GPU加速。



![05matrix](https://i.loli.net/2018/12/19/5c19c42852b5e.png)

总的来说



![06output](https://i.loli.net/2018/12/19/5c19c428534fb.png)

整个Network可以看成是 Feature extractor replacing feature engineering。在Hidden Layers中不断地进行特征抽取。使得特征能够被分类开来，在最后进行一个分类，得到output。



**Output Layer**可以看成是分类器，也可以用到 Softmax 函数。



---

> 例子：数字识别



输入是一个手写的数字（Image），输入是数字。



![07example](https://i.loli.net/2018/12/19/5c19c42853764.png)



输入的图片是 16(pixel)*16(pixel)。可以看成是一个256维的列向量，等于将图片中每一格像素涂黑表示1，不涂黑表示0（简化），然后从右往左，从上往下数，记录到列向量中。





output如果用的是 Softmax的话，可以代表几率。如果是10维的话，可以表示结果为对应到每一个数字的几率。



假如输出2的几率最大就输出2。



**要做到这件事，首先需要的是一个Function(Neural Network)**，将输入(256维向量)到输出(10维向量)。

![08example](https://i.loli.net/2018/12/19/5c19c42853910.png)

![09example](https://i.loli.net/2018/12/19/5c19c42853698.png)



结构自定。输入是256维向量，输出是10维，中间有几个Hidden Layers，每个Hidden Layers分别有几个neuron都是要自己去设计的。如果这个结构不好，就好比是大海捞针结果发现针不在海里，无法找到最好的Function。



设计需要靠 Trial and error 和 Intuition。具体的方法目前超越了人类的认知。有一种方法叫：**Evolutionary Artificial Neural Networks**，但是目前还是没有非常普及，效果也不是特别显著



除了**Fully Connect Feedforward Network**，还可以有很多特殊的接法，例如**Convolutional Neural Network(CNN)**





**接下来要做的就是要找到这个具体的Function**，可以用**Gradient Descent**





**Loss for an Example**

> Cross Entropy

在某一个数据中，通过Softmax函数输出的是几率，和前面一样，先设一组\\(\hat{y}\\)，例如：1、2、3 可以分别表示如下：


$$
\begin{bmatrix}
1 \\
0\\
0\\
0\\
0\\
0\\
0\\
0\\
0\\
0\\

\end{bmatrix} \qquad
\begin{bmatrix}
0 \\
1\\
0\\
0\\
0\\
0\\
0\\
0\\
0\\
0\\

\end{bmatrix}\qquad
\begin{bmatrix}
0 \\
0\\
1\\
0\\
0\\
0\\
0\\
0\\
0\\
0\\

\end{bmatrix}
$$


计算 Cross Entropy ：


$$
l(y,\hat{y})=C(y,\hat{y})=-\sum_{i=1}^{10}\hat{y}_i \ln y_i
$$


这里反应的是整体，其中:\\(\hat{y}\\)表示（后面的项）对某一项生效，
$$
\ln y_i
$$
表示生成某一项的可能性，如果可能性为1，则整体为0。





Cross Entropy 表示两个分布的相似程度，越小越好。若两个分布一模一样，则 Cross Entropy 为0。



![10CrossEntropy](https://i.loli.net/2018/12/19/5c19c42853093.png)



Target (即\\(\hat{y}\\))是一个10维的Vector。





**对于所有Training data 而言**



![11CrossEntropy](https://i.loli.net/2018/12/19/5c19c4ae11efa.png)





将所有 Cross Entropy 加起来，得到一个总的 Loos(Total Loss)：


$$
L=\sum^N_{n=1}l^n
$$



**接下来要做的就是要在 Function Set 中找到一个Function，能够最小化 Total Loss L，或者说找到神经网络中的各个参数，能够最小化 Total Loss L。**



**Gradient Descent**





方法同前面**Linear Regression**的一样。

1. 定初始值
2. 计算偏微分
3. 设置Learning-rate
4. 更新参数
5. 重复





![12GD](https://i.loli.net/2018/12/19/5c19c4ae0dc30.png)

![13GD](https://i.loli.net/2018/12/19/5c19c4ae13cbe.png)



> 计算偏微分这一步，可能有数以万计的参数，如果手动算会非常非常复杂，可以用到Backpropagation和很多的Tool kit。

![14tool kit](https://i.loli.net/2018/12/19/5c19c4ae0fd98.png)





## Why Deep?

