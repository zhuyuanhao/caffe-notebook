`Blob`可以看作是两个维度相同的多维数组的组合(老版本中固定为4维，代表num,channel,width,height)，两个多维数组分别对应`data`域和`diff`域。`data`域和`diff`域分别各使用一个`SyncedMemory`对象来保存数据，`SyncedMemory`本身是一维的`void*`数组，`Blob`通过内部函数实现下标的转换。

# 文件
```
include/caffe/blob.hpp
src/caffe/blob.cpp
```

# 依赖
1. 使用`SyncedMemory`保存实际的数据
2. 使用`share_ptr`管理`SyncedMemory`对象，所以`Blob`没有自己实现析构函数
3. 计算统计值时会用到对应CPU或GPU端的`blas`函数完成，函数定义在`include/caffe/util/math_functions.hpp`
4. 通过`src/caffe/proto/caffe.proto`文件中定义的`BlobProto`类来完成`Blob`的存储。         

```protobuf
message BlobShape {
  repeated int64 dim = 1 [packed = true];
}

message BlobProto {
  optional BlobShape shape = 7; 
  repeated float data = 5 [packed = true];
  repeated float diff = 6 [packed = true];
  repeated double double_data = 8 [packed = true];
  repeated double double_diff = 9 [packed = true];

  optional int32 num = 1 [default = 0];
  optional int32 channels = 2 [default = 0];
  optional int32 height = 3 [default = 0];
  optional int32 width = 4 [default = 0];
}

message BlobProtoVector {
  repeated BlobProto blobs = 1; 
}
```

# 成员
```cpp
template <typename Dtype>
class Blob {
 public:
  // 构造函数，没有析构函数
  Blob();
  explicit Blob(const int num, const int channels, const int height, const int width);
  explicit Blob(const vector<int>& shape);

  // 修改blob的维度参数，并在需要时重新分配适合大小的SyncedMemory。注意新分配的SyncedMemory内部是没有分配内存的
  void Reshape(const int num, const int channels, const int height,
      const int width);
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);

  // 查询总维度大小,如(64,3,28,28),或某一维的大小，支持负数作索引
  inline string shape_string() const;
  inline const vector<int>& shape() const;
  const int* gpu_shape() const;                         // 返回表示维度大小的数组在GPU上的指针，即shape_data_->gpu_data()
  inline int shape(int index) const;
  inline int CanonicalAxisIndex(int axis_index) const;  // 将正索引或负索引统一转为正索引。例如共4维时，[-4,3] -> [0,3]
  
  // 查询维度，如4维
  inline int num_axes() const { return shape_.size(); }

  // 统计全部或区间内元素个数
  inline int count() const;
  inline int count(int start_axis, int end_axis) const;
  inline int count(int start_axis) const;
  
  // 老版本的维度访问函数
  inline int num() const { return LegacyShape(0); }
  inline int channels() const { return LegacyShape(1); }
  inline int height() const { return LegacyShape(2); }
  inline int width() const { return LegacyShape(3); }
  inline int LegacyShape(int index) const;

  // 计算blob的多维下标在实际SyncedMemory中的位置
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const;
  inline int offset(const vector<int>& indices) const;

  // 拷贝其他blob的data或者diff数据，通过copy_diff参数判断。如果源blob和自己维度不一致，需要设reshape参数为true
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  // 查询多位数组中的某个元素的data或diff值
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const;
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const;
  inline Dtype data_at(const vector<int>& index) const;
  inline Dtype diff_at(const vector<int>& index) const;

  // 获取内部SyncedMemory的指针
  inline const shared_ptr<SyncedMemory>& data() const;
  inline const shared_ptr<SyncedMemory>& diff() const;

  // 返回或设置data多维数组的指针
  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const Dtype* gpu_data() const;
  void set_gpu_data(Dtype* data);
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();

  // 返回diff多维数组的指针
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();

  // 更新data多维数组，即计算data_ -= diff_
  void Update();

  // 根据Protobuf中的blob数据保存或恢复到当前blob对象。blob在Protobuf中的保存为BlobProto类型，BlobProto只支持float或double类型的blob
  void FromProto(const BlobProto& proto, bool reshape = true);
  void ToProto(BlobProto* proto, bool write_diff = false) const;
  bool ShapeEquals(const BlobProto& other);

  // 计算data数组或diff数组的L1正则化值,absolute sum，即sum(|x_i|)
  Dtype asum_data() const;
  Dtype asum_diff() const;
  // 计算data数组或diff数组的L2正则化值,sum square，即sum(x_i ^ 2)
  Dtype sumsq_data() const;
  Dtype sumsq_diff() const;

  // 将data或diff的数据更新乘以一个常量值
  void scale_data(Dtype scale_factor);
  void scale_diff(Dtype scale_factor);

  // 使用other的data或diff多维数组作为自己的数据，只修改自己的data_或diff_指针
  void ShareData(const Blob& other);
  void ShareDiff(const Blob& other);

 protected:
  shared_ptr<SyncedMemory> data_;               // data多维数组
  shared_ptr<SyncedMemory> diff_;               // diff多维数组
  shared_ptr<SyncedMemory> shape_data_;         // 使用SyncedMemory保存维度大小，这样在GPU端也能快速访问
  vector<int> shape_;                           // 维度大小，用于CPU端访问
  int count_;                                   // 多维数组的元素个数
  int capacity_;                                // 实际的SyncedMemory的容量

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob
```

# 实现要点
1. 初始化函数实际使用Reshape去设置内部变量，注意此时内部SyncedMemory的内部是没有分配内存的，需要调用对应内存访问函数后才会分配内存。
2. 在`set_cpu_data`或`set_gpu_data`的实现中，如果内部`data_`是`reshape()`变小过的，则传入的数据块的大小比内部`data_`的长度要小，所以需要重设`data_`和`diff_`，以防止`data_`在cpu和gpu端的内存块大小不一致而导致的同步错误。
```cpp
template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data) {
  CHECK(data);
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_gpu_data(data);
}
```
3. `Update, *Proto*, *sum/scale*`函数都不允许使用`int/unsigned`版本，只实例化了`float/double`版本
