# 文件
```
include/caffe/layers/prelu_layer.hpp
src/caffe/layers/prelu_layer.cpp
src/caffe/layers/prelu_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional PReLUParameter prelu_param = 131;
}

message PReLUParameter {
  optional FillerParameter filler = 1;
  optional bool channel_shared = 2 [default = false];
}
```
