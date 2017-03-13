`SPPLayer`用于空间金字塔池化（Spatial Pyramid Pooling）。SPP其实就是一种多个scale的pooling，可以获取图像中的多尺度信息；在CNN中加入SPP后，可以让CNN处理任意大小的输入，这让模型变得更加的flexible。

# 文件
```
include/caffe/layers/spp_layer.hpp
src/caffe/layers/spp_layer.cpp
```

# 参数
```protobuf
optional SPPParameter spp_param = 132;

message SPPParameter {
  enum PoolMethod {
    MAX = 0; 
    AVE = 1; 
    STOCHASTIC = 2; 
  }
  optional uint32 pyramid_height = 1; 
  optional PoolMethod pool = 2 [default = MAX];
  enum Engine {
    DEFAULT = 0; 
    CAFFE = 1; 
    CUDNN = 2; 
  }
  optional Engine engine = 6 [default = DEFAULT];
}
```