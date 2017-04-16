RMSprop可以算作Adadelta的一个特例：

当$$\rho=0.5$$ 时，$$E|g^2|_t=\rho*E|g^2|_{t-1}+(1-\rho)*g_t^2$$ 就变为了求梯度平方和的平均数。

如果再求根的话，就变成了RMS(均方根)：
$$RMS|g|_t=\sqrt{E|g^2|_t+\epsilon}$$

此时，这个RMS就可以作为学习率$$\eta$$的一个约束：
$$\Delta{x_t}=-\frac{\eta}{RMS|g|_t}*g_t$$

特点：
* 其实RMSprop依然依赖于全局学习率
* RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间
* 适合处理非平稳目标 - 对于RNN效果很好