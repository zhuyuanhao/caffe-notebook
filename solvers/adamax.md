Adamax是Adam的一种变体，此方法对学习率的上限提供了一个更简单的范围。公式上的变化如下：
$$n_t=max(\nu*n_{t-1},|g_t|)$$
$$\Delta{x}=-\frac{\hat{m_t}}{n_t+\epsilon}*\eta$$

可以看出，Adamax学习率的边界范围更简单