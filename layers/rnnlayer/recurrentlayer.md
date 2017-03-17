`RecurrentLayer`作为所有循环神经网络的基类层，具体的实现由`RNNLayer`和`LSTMLayer`完成。

# 文件
```
include/caffe/layers/recurrent_layer.hpp
src/caffe/layers/recurrent_layer.cpp
src/caffe/layers/recurrent_layer.cu
```

# 原理

# 参数
```protobuf
optional RecurrentParameter recurrent_param = 146;


```