Adadelta是对Adagrad的扩展，最初方案依然是对学习率进行自适应约束，但是进行了计算上的简化。 Adagrad会累加之前所有的梯度平方，而Adadelta只累加固定大小的项，并且也不直接存储这些项，仅仅是近似计算对应的平均值。即：
$$n_t=\nu*n_{t-1}+(1-\nu)*g_t^2$$
$$\Delta{\theta_t} = -\frac{\eta}{\sqrt{n_t+\epsilon}}*g_t$$
在此处Adadelta其实还是依赖于全局学习率的，但是作者做了一定处理，经过近似牛顿迭代法之后：
$$E|g^2|_t=\rho*E|g^2|_{t-1}+(1-\rho)*g_t^2$$
$$\Delta{x_t}=-\frac{\sqrt{\sum_{r=1}^{t-1}\Delta{x_r}}}{\sqrt{E|g^2|_t+\epsilon}}$$
其中，E代表求期望。

此时，可以看出Adadelta已经不用依赖于全局学习率了。

特点：
* 训练初中期，加速效果不错，很快
* 训练后期，反复在局部最小值附近抖动