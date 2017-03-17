# 文件
```
include/caffe/layers/elu_layer.hpp
src/caffe/layers/elu_layer.cpp
src/caffe/layers/elu_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional ELUParameter elu_param = 140;
}

message ELUParameter {
  optional float alpha = 1 [default = 1];
}
```
