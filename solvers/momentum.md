momentum是模拟物理里动量的概念，积累之前的动量来替代真正的梯度。公式如下：

$$m_t=\mu*m_{t-1}+g_t$$
$$\Delta{\theta_t}=-\eta*m_t$$
其中，$$\mu$$是动量因子

特点：
* 下降初期时，使用上一次参数更新，下降方向一致，乘上较大的$$\mu$$ 能够进行很好的加速
* 下降中后期时，在局部最小值来回震荡的时候，$$gradient\to0$$，$$\mu$$ 使得更新幅度增大，跳出陷阱
* 在梯度改变方向的时候，$$\mu$$ 能够减少更新 总而言之，momentum项能够在相关方向加速SGD，抑制振荡，从而加快收敛