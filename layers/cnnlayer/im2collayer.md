`Im2colLayer`用于早期Caffe中实现卷积操作时将image转换成列矩阵的操作，现在这一操作已经集成到`BaseConvolutionLayer`中。`Im2colLayer`使用`ConvolutionParameter`做参数。

# 文件
```
include/caffe/layers/im2col_layer.hpp
src/caffe/layers/im2col_layer.cpp
src/caffe/layers/im2col_layer.cu
```
