# 文件
```
Header: ./include/caffe/layers/dummy_data_layer.hpp
CPU: ./src/caffe/layers/dummy_data_layer.cpp
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional DummyDataParameter dummy_data_param = 109;
}

// 使用Filler填充任意维的blob作为数据输入层
message DummyDataParameter {
  // filler个数可以为0个、1个或和shape个数相同, 如果是0个，则都用常数0的ConstantFiller填充
  // 如果是1个，则所有输出blob都用这个filler填充，和shape个数相同时分别用对应filler填充
  repeated FillerParameter data_filler = 1; // filler个数和shape个数对应，如果
  repeated BlobShape shape = 6;             // shape个数对应输出blob的个数

  // 已废弃，建议用shape参数
  repeated uint32 num = 2; 
  repeated uint32 channels = 3; 
  repeated uint32 height = 4; 
  repeated uint32 width = 5; 
}
```

