# 文件
```
include/caffe/layers/power_layer.hpp
src/caffe/layers/power_layer.cpp
src/caffe/layers/power_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional PowerParameter power_param = 122;
}

message PowerParameter {
  optional float power = 1 [default = 1.0];
  optional float scale = 2 [default = 1.0];
  optional float shift = 3 [default = 0.0];
}
```
