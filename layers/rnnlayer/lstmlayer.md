`LSTMLayer`继承自`RecurrentLayer`。

# 文件
```
include/caffe/layers/lstm_layer.hpp
src/caffe/layers/lstm_layer.cpp
src/caffe/layers/lstm_unit_layer.cpp
src/caffe/layers/lstm_unit_layer.cu
```

# 原理

# 参数
使用和RecurrentLayer一样的参数。
```protobuf
optional RecurrentParameter recurrent_param = 146;
```