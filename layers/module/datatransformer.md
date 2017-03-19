`DataTransformer`类用于输入图片的预处理，在`LayerParameter`中包含了相关字段，但只在数据读取层中被使用。

# 文件
```
include/caffe/data_transformer.hpp
src/caffe/data_transformer.cpp
```
# 依赖
2. `Datum`类用于存储图片的序列化表示，用于`lmdb/leveldb`的数据读取和保存
```protobuf
// lmdb/leveldb中的图片存储格式，一张图片一个Datum
message Datum {
  optional int32 channels = 1;                  // 通道数 
  optional int32 height = 2;                    // 高度
  optional int32 width = 3;                     // 宽度
  optional bytes data = 4;                      // 实际的数据，以字节数组格式存储
  optional int32 label = 5;                     // 实际的label，32位整数
  repeated float float_data = 6;                // 也可以保存float格式的数据
  optional bool encoded = 7 [default = false];  // data/float_data是否时编码的数据
}
``` 
1. 数据转换的参数出现在`LayerParameter`中，使用转换参数类`TransformationParameter`

```protobuf
message LayerParameter {
  ...
  optional TransformationParameter transform_param = 100;   // 数据读取层的预处理参数类
  ...
}
message TransformationParameter {                   // 数据读取层的预处理参数类
  optional float scale = 1 [default = 1];           // 元素值按比例增减
  optional bool mirror = 2 [default = false];       // 是否作随机镜像变换
  optional uint32 crop_size = 3 [default = 0];      // 随机位置截取crop_size*crop_size大小的子图，默认用全图
  optional string mean_file = 4;                    // 均值所在文件
  repeated float mean_value = 5;                    // 均值，不能和mean_file同时出现。可以为1个，或channels个
  optional bool force_color = 6 [default = false];  // 是否强制生成3通道数据
  optional bool force_gray = 7 [default = false];   // 是否强制生成1通道数据
}
```

# 成员
```cpp
// 输入图片的预处理类
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);  // Train和Test的预处理稍有不同
  virtual ~DataTransformer() {}

  // 不同类型源数据的预处理函数，最后都写到blob中。目标blob可以共享top blob的data数据，这样可以减少数据拷贝
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);
  void Transform(const vector<Datum> & datum_vector, Blob<Dtype>* transformed_blob);
#ifdef USE_OPENCV
  void Transform(const vector<cv::Mat> & mat_vector, Blob<Dtype>* transformed_blob);
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);
#endif  // USE_OPENCV
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  // 推断预处理后的blob的维度，一张图片时，num维度为1
  vector<int> InferBlobShape(const Datum& datum);
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
#ifdef USE_OPENCV
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

  // 初始化随机数产生器
  void InitRand();
 protected:
  // 随机数产生器：返回[0, n-1]的随机数
  virtual int Rand(int n);
  void Transform(const Datum& datum, Dtype* transformed_data);

  TransformationParameter param_;               // 预处理参数
  shared_ptr<Caffe::RNG> rng_;                  // 内部随机数产生器
  Phase phase_;                                 // 网络状态
  Blob<Dtype> data_mean_;                       // 从mean_file中读取的均值blob
  vector<Dtype> mean_values_;                   // 均值数组，可以只有一个或通道数个
};
```
