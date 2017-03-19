Filler用于为Blob填充数据，但只填充Blob的CPU data数组的数据

# 文件
```
include/caffe/filler.hpp
```

# 原理

# 参数
```protobuf
message FillerParameter {
  optional string type = 1 [default = 'constant'];  // 字符串作类型
  optional float value = 2 [default = 0];           // 常量填充器的参数值
  optional float min = 3 [default = 0];             // uniform填充器的最小值参数
  optional float max = 4 [default = 1];             // uniform填充器的最大值参数
  optional float mean = 5 [default = 0];            // Gaussian填充器的均值参数
  optional float std = 6 [default = 1];             // Gaussian填充器的标准差参数
  optional int32 sparse = 7 [default = -1];         // Gaussian填充器的参数
  enum VarianceNorm {
    FAN_IN = 0; 
    FAN_OUT = 1; 
    AVERAGE = 2; 
  }
  optional VarianceNorm variance_norm = 8 [default = FAN_IN];   // xavier,msra填充器的参数
}
```