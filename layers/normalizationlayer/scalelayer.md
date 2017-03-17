将输入的第二个blob的数据按元素乘到第一个blob中。


# 文件
```
include/caffe/layers/scale_layer.hpp
src/caffe/layers/scale_layer.cpp
src/caffe/layers/scale_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional ScaleParameter scale_param = 142;
}

message ScaleParameter {
  optional int32 axis = 1 [default = 1];
  optional int32 num_axes = 2 [default = 1];
  optional FillerParameter filler = 3;
  optional bool bias_term = 4 [default = false];
  optional FillerParameter bias_filler = 5;
}
```