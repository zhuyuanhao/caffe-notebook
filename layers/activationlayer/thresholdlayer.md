# 文件
```
Header: ./include/caffe/layers/threshold_layer.hpp
CPU: ./src/caffe/layers/threshold_layer.cpp
CUDA GPU: ./src/caffe/layers/threshold_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional ThresholdParameter threshold_param = 128;
}

message ThresholdParameter {
  optional float threshold = 1 [default = 0];
}
```
