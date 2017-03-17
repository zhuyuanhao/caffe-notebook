# 文件
```
include/caffe/layers/batch_norm_layer.hpp
src/caffe/layers/batch_norm_layer.cpp
src/caffe/layers/batch_norm_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional BatchNormParameter batch_norm_param = 139;
}

message BatchNormParameter {
  optional bool use_global_stats = 1;
  optional float moving_average_fraction = 2 [default = .999];
  optional float eps = 3 [default = 1e-5];
}
```