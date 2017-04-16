nesterov项在梯度更新时做一个校正，避免前进太快，同时提高灵敏度。 将Momentum 中的公式展开可得：
$$\Delta{\theta_t}=-\eta*\mu*m_{t-1}-\eta*g_t$$
可以看出，$$m_{t-1}$$ 并没有直接改变当前梯度$$g_t$$。

Nesterov的改进就是让之前的动量直接影响当前的动量。即：

$$g_t=\nabla_{\theta_{t-1}}{f(\theta_{t-1}-\eta*\mu*m_{t-1})}$$
$$m_t=\mu*m_{t-1}+g_t$$
$$\Delta{\theta_t}=-\eta*m_t$$
所以，加上nesterov项后，梯度在大的跳跃后，进行计算对当前梯度进行校正。如下图：