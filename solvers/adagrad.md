Adagrad其实是对学习率进行了一个约束。即：

$$n_t=n_{t-1}+g_t^2$$
$$\Delta{\theta_t}=-\frac{\eta}{\sqrt{n_t+\epsilon}}*g_t$$
此处，对$$g_t$$从1到t进行一个递推形成一个约束项regularizer：$$-\frac{1}{\sqrt{\sum_{r=1}^t(g_r)^2+\epsilon}}$$，$$\epsilon$$ 用来保证分母非0

特点：
* 前期$$g_t$$ 较小的时候， regularizer较大，能够放大梯度
* 后期$$g_t$$ 较大的时候，regularizer较小，能够约束梯度
* 适合处理稀疏梯度

缺点：
* 由公式可以看出，仍依赖于人工设置一个全局学习率
* $$\eta$$ 设置过大的话，会使regularizer过于敏感，对梯度的调节太大
* 中后期，分母上梯度平方的累加将会越来越大，使$$gradient\to0$$，使得训练提前结束