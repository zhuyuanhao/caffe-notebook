# 文件
```
include/caffe/layers/batch_norm_layer.hpp
src/caffe/layers/batch_norm_layer.cpp
src/caffe/layers/batch_norm_layer.cu
```

# 原理
1. 对每个Channel，计算Batch内的所有图片的同Channel数据的mean和variance。
2. 内部blobs_用于保存mean, variance和moving average factor，并设置learning rate为0，防止BP的时候修改。mean和variance有Channel个，moving average factor只有一个。


# 参数
```protobuf
message LayerParameter {
  optional BatchNormParameter batch_norm_param = 139;
}

message BatchNormParameter {
  optional bool use_global_stats = 1;
  optional float moving_average_fraction = 2 [default = .999];
  optional float eps = 3 [default = 1e-5];
}
```