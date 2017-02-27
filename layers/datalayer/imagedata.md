`ImageDataLayer`继承自`BasePrefetchingDataLayer`，用于从文件列表中读取图片数据。

# 文件
```
include/caffe/layers/image_data_layer.hpp
src/caffe/layers/image_data_layer.cpp
```
# 依赖
1. 在Layer中使用`ImageDataParameter image_data_param`作为参数
```protobuf
message ImageDataParameter {
  optional string source = 1;                   // 图片列表文件
  optional uint32 batch_size = 4 [default = 1]; // batch size
  optional uint32 rand_skip = 7 [default = 0];  // 用于在asgd的时候不同的client从(0, rand_skip)的位置开始读取数据
  optional bool shuffle = 8 [default = false];  // 每个epoch是否随机化
  optional uint32 new_height = 9 [default = 0]; // 不为0时，图片会resize到指定大小
  optional uint32 new_width = 10 [default = 0];
  optional bool is_color = 11 [default = true]; // 指定图片是否时彩色的
  optional float scale = 2 [default = 1];       // 已废弃，用transformer来处理
  optional string mean_file = 3;                // 已废弃，用transformer来处理
  optional uint32 crop_size = 5 [default = 0];  // 已废弃，用transformer来处理
  optional bool mirror = 6 [default = false];   // 已废弃，用transformer来处理
  optional string root_folder = 12 [default = ""];// 图片数据的根目录
}
```
# 对象
```cpp
template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }   // 指定类型
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;                             // 随机化图片列表的随机数产生器
  virtual void ShuffleImages();                                     // 通过随机化图片文件列表来随机化读取数据
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, int> > lines_;                      // 所有图片文件的名称和label
  int lines_id_;                                                    // 当前读取的文件序号
};
```