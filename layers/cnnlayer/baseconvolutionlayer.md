`BaseConvolutionLayer`作为卷积操作的类，被`ConvolutionLayer`和`DeconvolutionLayer`继承。卷积在计算中总是先把图片和卷积都转成矩阵，使用矩阵相乘就得到最后的数据。将`C*K*K`维卷积核转换成卷积核矩阵中的一行数据，行数根据图片矩阵计算，矩阵中每行数据相同。将`C*H*W`维图片中的`C*K*K`大的子区域转换成图片矩阵中的一行数据，根据步长每行取不同的数据，最后结果是卷积核矩阵乘以图片矩阵的转置。`BaseConvolutionLayer`提供了卷积相关的基本操作。

# 文件
```
include/caffe/layers/base_conv_layer.hpp
src/caffe/layers/base_conv_layer.cpp
```

# 原理
1. `group convolution`用于模拟`Alex Krizhevsky`最初的论文中的模型，其他地方没啥用。当`group=2`时，前一半的filters只和输入数据的前一半channel做卷积，后一半filters只和输入数据的后一半channel做卷积。

# 参数
1. `BaseConvolutionLayer`，`ConvolutionLayer`和`DeconvolutionLayer`都是使用`ConvolutionParameter`做参数

```protobuf
message LayerParameter {
  optional ConvolutionParameter convolution_param = 106;
}

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

message ConvolutionParameter {
  optional uint32 num_output = 1;               // 输出map个数
  optional bool bias_term = 2 [default = true]; // 是否包含bias

  // pad，kernel，stride只有一个数字时，认为height和width相等
  repeated uint32 pad = 3;                      // 边缘补齐维度，默认为0
  repeated uint32 kernel_size = 4;              // 卷积核维度，可以只有1个或和图片的维度数相同个
  repeated uint32 stride = 6;                   // 步长维度，默认为1
  repeated uint32 dilation = 18;                // 膨胀卷积(dilated convolution)参数，默认为1

  // 二维卷积时的参数
  optional uint32 pad_h = 9 [default = 0];
  optional uint32 pad_w = 10 [default = 0];
  optional uint32 kernel_h = 11;
  optional uint32 kernel_w = 12;
  optional uint32 stride_h = 13;
  optional uint32 stride_w = 14;

  optional uint32 group = 5 [default = 1];      // group convolution时的组大小

  optional FillerParameter weight_filler = 7;   // 权重weight的填充器
  optional FillerParameter bias_filler = 8;     // 偏置bias的填充器
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 15 [default = DEFAULT];

  // channel维度的下标，比如输入是(N, C, D, H, W)，axis==1表示以
  // (D, H, W)作输入图片，和(C/group)个通道的卷积核作N次卷积
  optional int32 axis = 16 [default = 1];
  // 是否强制使用普通的N维im2col，目前2维的有特殊实现
  optional bool force_nd_im2col = 17 [default = false];
}
```