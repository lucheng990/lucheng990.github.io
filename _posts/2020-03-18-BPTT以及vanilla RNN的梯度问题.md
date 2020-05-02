```
tag: 机器学习
```







通常，全连接RNN（fully connected RNN）会存在梯度消失（graident vanishing）和梯度爆炸（gradient exploding)的问题。



用Elman network为例，其formulation：


$$
\begin{align*}
& h^t = \sigma_h(W_hx^t + U_hh^{t-1} +b_h) \tag{1} \\
& y^t = \sigma_y(W_yh^t+ b_y)    \tag{2}
\end{align*}
$$




其中，\\(x^t\\)为输入的data x，\\(h^t\\)是hidden layer output，\\(y^t\\)为最后的输出；\\(W,U,b\\)是需要待训练的参数（params）；\\(\sigma\\)为激活函数。





和传统的NN在训练时候有点区别，训练Elman Network，或者是RNN，需要用到BPTT算法，



## BPTT



因为RNN的缘故，每次输入都是一个时间序列，在每一个时间序列（time step）中的每一个小时间点t都有一个输入，这些输入共享weights然后输出。更新参数的时候，是**每个time step更新一次**，在一个time step（一个sequence）中，假设有T个小段t，就会得到T个输出：


$$
y^1, y^2,y^3,\cdots ,y^T
$$


根据标签可以得到一个时间序列的总的Loss：


$$
L =  \sum_{t=1}^T l^t=\sum_{t=1}^T {l(\hat{y}^t,y^t)}   \tag{3}
$$



有了这个loss function，目的是要来更新参数。**怎么更新参数？**更新参数的梯度来自于误差；**误差来自于哪里？**误差来自于backward pass，从后面（深层）往前传。**但是，如果是普通的NN，直接从后（深层）往前传就行，RNN还需要考虑时间维度（through time），所以误差会来自于更深层以及下一个时间点两个方向。**误差根源是来自输出y和标签之间的关系。



要理解这一点，考虑forward pass：在t时刻有一个输入，输入会和这些参数做一些运算，得到在t时刻的输出，同时会将hidden layer信息保留到下一个时间点。换句话说，如果这些参数有一个微小的改动，那么会同时影响到该时刻的输出以及后面时刻的输出。所以后面梯度（误差）的传播也是要沿着这些路径来。





下面上两张图片（来源于[网址](https://www.cnblogs.com/yanshw/p/10478876.html)，[网址](https://www.youtube.com/watch?v=UTojaCzMoOs&list=PLlPcwHqLqJDkVO0zHMqswX1jA9Xw7OSOK&index=4&t=278s)）直观地显示参数对各个l的影响以及最后误差的传递过程，**这里的左右表示时间维度**：



![a1](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic/26RNN2/a1.png)



![a2](https://luc-website.oss-cn-hangzhou.aliyuncs.com/websitepic/26RNN2/a2.png)







根据forward pass和backward pass，对于时间t上面的\\(W_h\\)参数，除了会接收来自于t时刻的从loss function处获得的gradient，还会接收来自于所有t+1以后的时刻的从loss function处获得的gradient。一般化的，假设现在在时刻\\(t\\)，根据上面（1）（2）（3）式子，可以写出该时刻的参数\\(W_h\\)（这里先只考虑这一个最复杂的参数），对总的Loss的偏微分：


$$
\frac{\partial L}{\partial W_h} = \sum_{i=t}^T \frac{\partial l^i}{\partial W_h}
$$


这是一个可递归式子，假设T=t，也就是最后一层（时间维度）了，那么：


$$
\begin{align*}
& \frac{\partial L}{\partial W_h} =  \frac{\partial l^t}{\partial W_h}=\frac{\partial l^t}{\partial y^t} \cdot \frac{\partial y^t}{\partial h^t} \cdot \frac{\partial h^t}{\partial W_h} \\
= &\sum_{i=t}^t ((\frac{\partial l^t}{\partial y^t} \cdot \frac{\partial y^t}{\partial h^t}) \prod_{k=t+1}^i(\frac{\partial h^{k}}{\partial h^{k-1}})\cdot \frac{\partial h^t}{\partial W_h})
\end{align*}
$$




假设T=t+1，也就是说在这个时刻t后面还有一个输入：


$$
\begin{align*}
& \frac{\partial L}{\partial W_h} = \sum_{i=t}^{t+1} \frac{\partial l^i}{\partial W_h} = \frac{\partial l^t}{\partial W_h}+\frac{\partial l^{t+1}}{\partial W_h} \\
= & \frac{\partial l^t}{\partial y^t} \cdot \frac{\partial y^t}{\partial h^t} \cdot \frac{\partial h^t}{\partial W_h}+ \frac{\partial l^{t+1}}{\partial y^{t+1}} \cdot \frac{\partial y^{t+1}}{\partial h^{t+1}} \cdot \frac{\partial h^{t+1}}{\partial h^t} \cdot \frac{\partial h^t}{\partial W_h} \\
= & \sum_{i=t}^{t+1} ((\frac{\partial l^{i}}{\partial y^{i}} \cdot \frac{\partial y^{i}}{\partial h^{i}})\prod_{k=t+1}^i(\frac{\partial h^{k}}{\partial h^{k-1}})\cdot \frac{\partial h^t}{\partial W_h})
\end{align*}
$$




假设T=t+2：


$$
\begin{align*}
& \frac{\partial L}{\partial W_h} = \sum_{i=t}^{t+2} \frac{\partial l^i}{\partial W_h} = \frac{\partial l^t}{\partial W_h}+\frac{\partial l^{t+1}}{\partial W_h} +\frac{\partial l^{t+2}}{\partial W_h}\\
= & \frac{\partial l^t}{\partial y^t} \cdot \frac{\partial y^t}{\partial h^t} \cdot \frac{\partial h^t}{\partial W_h}+ \frac{\partial l^{t+1}}{\partial y^{t+1}} \cdot \frac{\partial y^{t+1}}{\partial h^{t+1}} \cdot \frac{\partial h^{t+1}}{\partial h^t} \cdot \frac{\partial h^t}{\partial W_h} + \frac{\partial l^{t+2}}{\partial y^{t+2}}\cdot \frac{\partial y^{t+2}}{h^{t+2}} \cdot \frac{\partial h^{t+2}}{\partial h^{t+1}} \cdot \frac{\partial h^{t+1}}{\partial h^t} \cdot \frac{\partial h^t}{\partial W_h} \\
= & \sum_{i=t}^{t+2}( (\frac{\partial l^{i}}{\partial y^{i}} \cdot \frac{\partial y^{i}}{\partial h^{i}})\prod_{k=t+1}^i(\frac{\partial h^{k}}{\partial h^{k-1}})\cdot \frac{\partial h^t}{\partial W_h})
\end{align*}
$$





推导到一个序列(time step)上来，注意上面的式子是针对任意一个时刻的，而在一整个序列中总共有T个时刻：




$$
\begin{align*}
& \frac{\partial L}{\partial W_h} = \sum_{t=1}^T \sum_{i=t}^{T} \frac{\partial l^i}{\partial W_h}  \\
= & \sum_{t=1}^T \sum_{i=t}^{T}( (\frac{\partial l^{i}}{\partial y^{i}} \cdot \frac{\partial y^{i}}{\partial h^{i}})\prod_{k=t+1}^i(\frac{\partial h^{k}}{\partial h^{k-1}})\cdot \frac{\partial h^t}{\partial W_h}) \tag{4}
\end{align*}
$$


（4）式里的各个因子都是可以求得（里面的变量要么通过forward pass得到，要么通过backward pass可以得到）









用同样的方法可以求得参数\\(U_h\\)对Loss的偏微分。事实上它和上面的\\(W_h\\)地位差不多，算法一致，包括后面的bias \\(b_n\\)，结果如下：


$$
\begin{align*}
& \frac{\partial L}{\partial U_h} = \sum_{t=1}^T \sum_{i=t}^{T} \frac{\partial l^i}{\partial U_h}  \\
= & \sum_{t=1}^T \sum_{i=t}^{T}( (\frac{\partial l^{i}}{\partial y^{i}} \cdot \frac{\partial y^{i}}{\partial h^{i}})\prod_{k=t+1}^i(\frac{\partial h^{k}}{\partial h^{k-1}})\cdot \frac{\partial h^t}{\partial U_h}) \tag{5}
\end{align*}
$$




$$
\begin{align*}
& \frac{\partial L}{\partial b_h} = \sum_{t=1}^T \sum_{i=t}^{T} \frac{\partial l^i}{\partial b_h}  \\
= & \sum_{t=1}^T \sum_{i=t}^{T}( (\frac{\partial l^{i}}{\partial y^{i}} \cdot \frac{\partial y^{i}}{\partial h^{i}})\prod_{k=t+1}^i(\frac{\partial h^{k}}{\partial h^{k-1}})\cdot \frac{\partial h^t}{\partial b_h}) \tag{6}
\end{align*}
$$





还有两个参数是\\(W_y\\)和\\(b_y\\)，这两个参数算法并不涉及through time，没有多路传递的gradients。算法如下：


$$
\frac{\partial L}{\partial W_y}= \sum_{t=1}^T \frac{\partial l^t}{\partial W_y}=\sum_{t=1}^T\frac{\partial l^t}{\partial y^t} \cdot \frac{\partial y^t}{\partial W_y}  \tag{7}
$$



$$
\frac{\partial L}{\partial b_y} = \sum_{t=1}^T \frac{\partial l^t}{\partial b_y}=\sum_{t=1}^T\frac{\partial l^t}{\partial y^t} \cdot \frac{\partial y^t}{\partial b_y} \tag{8}
$$

---



上面是BPTT的整个过程（在valina RNN）上面的实现，可以注意到在（4）（5）（6）式中有一项是连乘的，如果一个输入的序列长度很长，T够大的话，反向传播的时候会产生梯度exploding和vanishing的问题，例如：


$$
0.99^{1000}= 很小一个数
$$



这个问题具体体现在：RNN一般是用来处理一些时间序列相关问题，各个输入中会有context的关系，而如果梯度是连乘（指数级）向前传递的，那么在传递过程中要么衰减的厉害要么增大的厉害。如果是gradient exploding可以用gradient clipping的方法去address这个问题，但是gradient vanishing解决起来很困难。所以往往造成的结果是，后面的时刻的输入很难影响到前面的输入，这样训练起来效果会不好。







例如，在（4）这个式子中，有一项：


$$
\prod_{k=t+1}^i\frac{\partial h^{k}}{\partial h^{k-1}}
$$




根据（1）式：



$$
h^t = \sigma_h(W_hx^t + U_hh^{t-1} +b_h) \tag{1}
$$






对这个进行求导，




$$
\frac{\partial h^t}{\partial h^{t-1}} = \sigma_h'(W_hx^t + U_hh^{t-1} +b_h) \cdot whatever
$$


这里的\\(\sigma\\)通常是tanh函数，这个函数的导数也是介于0-1之间的，连乘以后会变得非常小。使用LSTM会对gradient vanishing问题有缓解。



---



参考资料：



[循环神经网络(二)-极其详细的推导BPTT](https://www.cnblogs.com/yanshw/p/10478876.html)

视频：

<iframe width="712" height="538" src="https://www.youtube.com/embed/UTojaCzMoOs?list=PLlPcwHqLqJDkVO0zHMqswX1jA9Xw7OSOK" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>





<iframe width="956" height="538" src="https://www.youtube.com/embed/3Hn_hEPtciQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>





