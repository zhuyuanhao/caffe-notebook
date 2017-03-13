`PoolingLayer`用于卷积中的下采样操作。

# 文件
```
include/caffe/layers/pooling_layer.hpp
src/caffe/layers/pooling_layer.cpp
src/caffe/layers/pooling_layer.cu
```

# 原理
`MAX Pooling`表示每次取子区块的最大值，`AVE Pooling`表示每次取子区块的平均值，`STOCHASTIC Pooling`表示按照子区块中的值的大小按概率取其中的某个值。

# 参数
```protobuf
message LayerParameter {
  optional PoolingParameter pooling_param = 121;
}

message PoolingParameter {
  enum PoolMethod {
    MAX = 0; 
    AVE = 1; 
    STOCHASTIC = 2; 
  }
  optional PoolMethod pool = 1 [default = MAX];
  optional uint32 pad = 4 [default = 0];
  optional uint32 pad_h = 9 [default = 0];
  optional uint32 pad_w = 10 [default = 0];
  optional uint32 kernel_size = 2; 
  optional uint32 kernel_h = 5; 
  optional uint32 kernel_w = 6; 
  optional uint32 stride = 3 [default = 1];
  optional uint32 stride_h = 7; 
  optional uint32 stride_w = 8; 
  enum Engine {
    DEFAULT = 0; 
    CAFFE = 1; 
    CUDNN = 2; 
  }
  optional Engine engine = 11 [default = DEFAULT];
  // 全局pooling时，kernel_h = bottom->height and kernel_w = bottom->width
  optional bool global_pooling = 12 [default = false];
}



```