# 文件
```
include/caffe/layers/exp_layer.hpp
src/caffe/layers/exp_layer.cpp
src/caffe/layers/exp_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional ExpParameter exp_param = 111;
}

message ExpParameter {
  optional float base = 1 [default = -1.0];
  optional float scale = 2 [default = 1.0];
  optional float shift = 3 [default = 0.0];
}
```
