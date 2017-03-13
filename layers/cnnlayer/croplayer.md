`CropLayer`的输入包含两个`blob`，将第一个`blob`按照第二个`blob`的维度做剪切后输出。

# 文件
```
include/caffe/layers/crop_layer.hpp
src/caffe/layers/crop_layer.cpp
src/caffe/layers/crop_layer.cu
```

# 参数
```protobuf
message LayerParameter {
  optional CropParameter crop_param = 144;
}

message CropParameter {
  optional int32 axis = 1 [default = 2]; // 图片开始的维度
  repeated uint32 offset = 2;            // 图片所有维或每一维的偏移
}
 ```