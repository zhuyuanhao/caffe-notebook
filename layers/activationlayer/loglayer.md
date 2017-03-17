# 文件
```
include/caffe/layers/log_layer.hpp
src/caffe/layers/log_layer.cpp
src/caffe/layers/log_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional LogParameter log_param = 134;
}

message LogParameter {
  optional float base = 1 [default = -1.0];
  optional float scale = 2 [default = 1.0];
  optional float shift = 3 [default = 0.0];
}
```
