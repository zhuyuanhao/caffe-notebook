MVN层将输入标准化为0-mean和1-variance

# 文件
```
include/caffe/layers/mvn_layer.hpp
src/caffe/layers/mvn_layer.cpp
src/caffe/layers/mvn_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
 optional MVNParameter mvn_param = 34;
}

message MVNParameter {
  optional bool normalize_variance = 1 [default = true];
  optional bool across_channels = 2 [default = false];
  optional float eps = 3 [default = 1e-9];
}

```