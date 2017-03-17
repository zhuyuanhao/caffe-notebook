全连接层一般用于最后的预测分类。

# 文件
```
include/caffe/layers/inner_product_layer.hpp
src/caffe/layers/inner_product_layer.cpp
src/caffe/layers/inner_product_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional InnerProductParameter inner_product_param = 17;
}

message InnerProductParameter {
  optional uint32 num_output = 1;
  optional bool bias_term = 2 [default = true];
  optional FillerParameter weight_filler = 3;
  optional FillerParameter bias_filler = 4;

  optional int32 axis = 5 [default = 1]; // 输入blob的feature map的起始维度
  optional bool transpose = 6 [default = false]; // 计算时是否使用weights矩阵的转置
}
```