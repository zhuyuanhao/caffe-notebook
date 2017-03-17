# 文件
```
include/caffe/layers/sigmoid_layer.hpp
src/caffe/layers/sigmoid_layer.cpp
src/caffe/layers/sigmoid_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional SigmoidParameter sigmoid_param = 124;
}

message SigmoidParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];
}
```
