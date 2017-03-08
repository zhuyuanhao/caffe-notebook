`BaseConvolutionLayer`作为卷积操作的类，被`ConvolutionLayer`和`DeconvolutionLayer`继承。卷积层在计算中总是先把多维图片和多维卷积核都转成二维矩阵，使用矩阵相乘得到最后的数据。`BaseConvolutionLayer`提供了卷积相关的基本操作。

# 文件
```
include/caffe/util/im2col.hpp
src/caffe/util/im2col.cpp
src/caffe/util/im2col.cu
include/caffe/layers/base_conv_layer.hpp
src/caffe/layers/base_conv_layer.cpp
```

# 原理
1. 卷积层可能有多个输入blob且每个blob有多张图片，但做卷积时每次只使用一张图片，产生一张图片，所有图片都使用相同的一组卷积核。对每张图片，输入数据、输出数据都是三维的blob，卷积核数组是四维的blob。其中
```cpp
kernel_channel == in_channel/group
kernel_num == out_channel
out_h == (in_h+2*pad_h-kernel_h)/stride_h+1
out_w == (in_w+2*pad_w-kernel_w)/stride_w+1
```
1. 计算一张图片的卷积时，使用`weight矩阵 * im2col矩阵`得到最终结果。
  * `weight矩阵`可以看作一个`[kernel_num, kernel_channel*kernel_h*kernel_w]`的矩阵，每行是一个完整的三维kernel的一维展开，weight blob中原本的排列方式就满足这种要求。
  * `im2col矩阵`是由一张`in_channel*height*width`的图片转成的一个维度为`[in_channel*kernel_h*kernel_w, kernel滑动总数]`的矩阵。其中每一列是一个原图三维子块的一维展开，大小等于group个kernel的元素数量和。列数是kernel滑动时产生的子块的总数，等于`out_h*out_w`。`im2col.hpp/.cpp`文件提供了单张图片和im2col矩阵之间的转换。
  * 最终的结果是一个`[out_channel, kernel滑动总数]`的矩阵，每行看作一个二维feature map的一维展开。
1. 计算`bias`时，`bias矩阵`的维度为`[kernel_num, 1]`，和一个全1的`[1, out_h*out_w]`矩阵相乘，把结果和输出矩阵做矩阵加。相当于第`i`个`feature map`的所有元素都加上`bias[i]`。
1. `group convolution`用于模拟`Alex Krizhevsky`最初的论文中的模型，其他地方没啥用。默认`group=1`，若`group>1`，输入输出的数据大小不变，由于`kernel_channel = in_channel/group`，所以内部weight参数的行数（卷积核数量）不变，列数（卷积核大小）除以group，im2col矩阵行数列数都不变。每张图片需要做`group`次卷积，每次使用一个group的卷积核和一个group的input channel。`weight`子矩阵的大小为`[kernel_num/group, kernel_channel*kernel_h*kernel_w]`，`im2col`子矩阵的大小为`[in_channel/group*kernel_h*kernel_w, kernel滑动总数]`。每个kernel只会和input的某一组channel做卷积，同一个group的kernel会和相同的这组channel做卷积，所以不同group的weights子矩阵和im2col子矩阵可以分布在不同的GPU上。
1. `dilated convolution`用于调整原图中做卷积的子块的元素间隔。当`dilation==2`时，原图子块是一个元素间间距为2的块。普通卷积所用的连续的`[channel, kernel_h, kernel_w]`的子块可以看做是`dilation==1`的实现。通过`kernel_h/w`计算`out_h/w`时，需要使用`kernel_h_with_dilation = dilation_h * (kernel_h - 1) + 1)`。

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

  optional uint32 group = 5 [default = 1];      // group convolution时的组数

  optional FillerParameter weight_filler = 7;   // 权重weight的填充器
  optional FillerParameter bias_filler = 8;     // 偏置bias的填充器
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 15 [default = DEFAULT];

  // channel维度的下标
  optional int32 axis = 16 [default = 1];
  // 是否强制使用普通的N维im2col，目前2维的有特殊实现
  optional bool force_nd_im2col = 17 [default = false];
}
```