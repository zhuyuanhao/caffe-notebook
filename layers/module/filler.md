Filler用于为Blob填充数据，可以是weight或者bias，但只填充Blob的CPU data数组的数据。

填充器种类包括：
* ConstantFiller：常量
* GaussianFiller：高斯分布（正态分布）
* UniformFiller：均匀分布
* PositiveUnitballFiller：先设为0-1的均匀分布，然后设置每一大行的和为1
* XavierFiller：
* MSRAFiller：
* BilinearFiller：双线性

它们都使用FillerParameter设置参数，不同的填充器使用不同的参数。
实现中，通过工厂方法`Filler<Dtype>* GetFiller(const FillerParameter& param);`获取相应的填充器实例。

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