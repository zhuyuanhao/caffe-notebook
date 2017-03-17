将输入的第二个blob的数据按元素加到第一个blob中。

# 文件
```
include/caffe/layers/bias_layer.hpp
src/caffe/layers/bias_layer.cpp
src/caffe/layers/bias_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional BiasParameter bias_param = 141;
}

message BiasParameter {
  optional int32 axis = 1 [default = 1];
  optional int32 num_axes = 2 [default = 1];
  optional FillerParameter filler = 3;
}
```