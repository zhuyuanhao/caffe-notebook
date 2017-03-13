`DeconvolutionLayer`继承自`BaseConvolutionLayer`，用于反卷积操作，可以看作是`ConvolutionLayer`的反向操作。

# 文件
```
include/caffe/layers/deconv_layer.hpp
src/caffe/layers/deconv_layer.cpp
src/caffe/layers/deconv_layer.cu
```
# 原理
反卷积和卷积的输入和输出是相反的。对于卷积公式`Y=W*X`，在卷积过程中是使用`W`和`X`求`Y`，在反卷积过程中是使用`W`和`Y`求`X`。在反卷积的`Forward`的过程中，使用卷积的`Backward`操作，同理反卷积的`Backward`过程中使用卷积的`Forward`操作。

# 参数
`ConvolutionLayer`和`DeconvolutionLayer`都是使用`ConvolutionParameter`做参数。