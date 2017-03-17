`DropoutLayer`用于在训练时将前一层传过来的数据随机置一部分为0，再传给下一层。

# 文件

```
include/caffe/layers/dropout_layer.hpp
src/caffe/layers/dropout_layer.cpp
src/caffe/layers/dropout_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional DropoutParameter dropout_param = 12;
}

message DropoutParameter {
  optional float dropout_ratio = 1 [default = 0.5]; // dropout ratio
}
```