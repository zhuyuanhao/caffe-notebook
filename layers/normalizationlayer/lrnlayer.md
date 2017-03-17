LRN全称为Local Response Normalization，即局部响应归一化层。

# 文件
```
include/caffe/layers/lrn_layer.hpp
src/caffe/layers/lrn_layer.cpp
src/caffe/layers/lrn_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional LRNParameter lrn_param = 118;
}

message LRNParameter {
  optional uint32 local_size = 1 [default = 5];
  optional float alpha = 2 [default = 1.]; 
  optional float beta = 3 [default = 0.75];
  enum NormRegion {
    ACROSS_CHANNELS = 0; 
    WITHIN_CHANNEL = 1; 
  }
  optional NormRegion norm_region = 4 [default = ACROSS_CHANNELS];
  optional float k = 5 [default = 1.]; 
  enum Engine {
    DEFAULT = 0; 
    CAFFE = 1; 
    CUDNN = 2; 
  }
  optional Engine engine = 6 [default = DEFAULT];
}
```