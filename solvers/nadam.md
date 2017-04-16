Nadam类似于带有Nesterov动量项的Adam。公式如下：
$$\hat{g_t}=\frac{g_t}{1-\Pi_{i=1}^t\mu_i}$$
$$m_t=\mu_t*m_{t-1}+(1-\mu_t)*g_t$$
$$\hat{m_t}=\frac{m_t}{1-\Pi_{i=1}^{t+1}\mu_i}$$
$$n_t=\nu*n_{t-1}+(1-\nu)*g_t^2$$
$$\hat{n_t}=\frac{n_t}{1-\nu^t}\bar{m_t}=(1-\mu_t)*\hat{g_t}+\mu_{t+1}*\hat{m_t}$$
$$\Delta{\theta_t}=-\eta*\frac{\bar{m_t}}{\sqrt{\hat{n_t}}+\epsilon}$$

可以看出，Nadam对学习率有了更强的约束，同时对梯度的更新也有更直接的影响。一般而言，在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果。