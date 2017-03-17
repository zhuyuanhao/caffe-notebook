# 文件
```
include/caffe/layers/relu_layer.hpp
src/caffe/layers/relu_layer.cpp
src/caffe/layers/relu_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional ReLUParameter relu_param = 123;
}

message ReLUParameter {
  optional float negative_slope = 1 [default = 0];
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 2 [default = DEFAULT];
}
```
