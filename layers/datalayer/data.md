`DataLayer`继承自`BasePrefetchingDataLayer`，用于从Database(LMDB/LevelDB)文件中读取数据。

# lmdb和leveldb
- 它们都是键/值对（Key/Value Pair）嵌入式数据库管理系统编程库。
- 虽然lmdb的内存消耗是leveldb的1.1倍，但是lmdb的速度比leveldb快10%至15%，更重要的是lmdb允许多种训练模型同时读取同一组数据集。
- 因此lmdb取代了leveldb成为Caffe默认的数据集生成格式。

# 文件
```

include/caffe/layers/data_layer.hpp
src/caffe/layers/data_layer.cpp
```

# 依赖
1. `DataLayer`使用`io.hpp`定义的`db::DB`接口和`db::Cursor`作为访问数据库文件的工具。数据库文件中的图片都是`Datum`格式，有些是编码后的数据，使用相应的`Datum`转图片的`OpenCV`函数处理。
1. 在`LayerParameter`中定义为`DataParameter data_param`

```protobuf
message DataParameter {
  enum DB { 
    LEVELDB = 0; 
    LMDB = 1; 
  }
  optional string source = 1;                   // database文件
  optional uint32 batch_size = 4;               // batch size
  optional uint32 rand_skip = 7 [default = 0];  // 已废弃，每个solver将访问database的不同子集
  optional DB backend = 8 [default = LEVELDB];  // db文件类型
  // DEPRECATED. See TransformationParameter. For data pre-processing, we can do
  // simple scaling and subtracting the data mean, if provided. Note that the
  // mean subtraction is always carried out before scaling.
  optional float scale = 2 [default = 1];       // 已废弃，用tranformer
  optional string mean_file = 3;                // 已废弃，用tranformer
  optional uint32 crop_size = 5 [default = 0];  // 已废弃，用tranformer
  optional bool mirror = 6 [default = false];   // 已废弃，用tranformer
  optional bool force_encoded_color = 9 [default = false];  // 是否将图片强制转换成彩色
  optional uint32 prefetch = 10 [default = 4];  // 预读取的batch数
}
```
# 成员
```cpp
template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; } // 数据并行时不共享
  virtual inline const char* type() const { return "Data"; }    // 类型是Data
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  void Next();                                                  // cursor_并不自动递增，需用Next()函数
  bool Skip();                                                  // 多solver时，每次找当前solver的图片
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<db::DB> db_;                                       // 存储当前db的实例
  shared_ptr<db::Cursor> cursor_;                               // 当前db实例的访问器
  uint64_t offset_;                                             // 图片数，总是递增的，不会重置
};
```
