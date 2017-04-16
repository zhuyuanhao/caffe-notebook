SGD指梯度下降法。一般有三种方式：
* batch gradient descent
* stochastic gradient descent
* mini-batch gradient descent

现在的SGD一般都指mini-batch gradient descent。

SGD就是每一次迭代计算mini-batch的梯度，然后对参数进行更新，是最常见的优化方法了。即：

$$g_t=\nabla_{\theta_{t-1}}{f(\theta_{t-1})}$$
$$\Delta{\theta_t}=-\eta*g_t$$
其中，$$\eta$$ 是学习率，$$g_t$$ 是梯度。

SGD完全依赖于当前batch的梯度，所以$$\eta$$可理解为允许当前batch的梯度多大程度影响参数更新

缺点：（正因为有这些缺点才让这么多大神发展出了后续的各种算法）
* 选择合适的learning rate比较困难
  对所有的参数更新使用同样的learning rate。对于稀疏数据或者特征，有时我们可能想更新快一些对于不经常出现的特征，对于常出现的特征更新慢一些，这时候SGD就不太能满足要求了
* SGD容易收敛到局部最优，并且在某些情况下可能被困在鞍点（经查阅论文发现，其实在合适的初始化和step size的情况下，鞍点的影响并没这么大）