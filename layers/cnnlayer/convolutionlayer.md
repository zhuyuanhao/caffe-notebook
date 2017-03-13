`ConvolutionLayer`继承自`BaseConvolutionLayer`，用于卷积操作。

# 文件
```
include/caffe/layers/conv_layer.hpp
src/caffe/layers/conv_layer.cpp
src/caffe/layers/conv_layer.cu
```
# 原理
`ConvolutionLayer`类使用基类`BaseConvolutionLayer`的工具函数完成`Forward/Backward`操作。
通过让函数`bool reverse_dimensions()`返回`false/true`，`Convolutuion`和`Deconvolution`层可以共用`BaseConvolutionLayer`的代码

# 参数
`ConvolutionLayer`和`DeconvolutionLayer`都是使用`ConvolutionParameter`做参数。