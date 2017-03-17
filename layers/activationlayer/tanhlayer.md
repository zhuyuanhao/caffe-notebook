# 文件
```
include/caffe/layers/tanh_layer.hpp
src/caffe/layers/tanh_layer.cpp
src/caffe/layers/tanh_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional TanHParameter tanh_param = 127;
}

message TanHParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];
}
```
