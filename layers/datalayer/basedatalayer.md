`basedata`定义了数据层的基类，包含`BaseDataLayer`和`BasePrefetchingDataLayer`，提供了以线程预取的方式读取数据的能力。

# 文件
```
include/caffe/internal_thread.hpp
src/caffe/internal_thread.cpp
include/caffe/util/blocking_queue.hpp
src/caffe/util/blocking_queue.cpp
include/caffe/layers/base_data_layer.hpp
src/caffe/layers/base_data_layer.cpp
```
# 依赖
1. `BaseDataLayer`继承自`Layer`，`BasePrefetchingDataLayer`继承自`BaseDataLayer`和`InternalThread`类。`BaseDataLayer`和`BasePrefetchingDataLayer`实现了`LayerSetUp()`函数，其他继承的layer自己实现`DataLayerSetUp()`函数。
2. 使用`BlockingQueue`来管理预读取的数据，数据读取线程不断检查是否有空闲的块可以拿来存储数据，主线程检查是否有已存储数据的块可以拿来计算。

# 成员
```cpp
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // 会调用DataLayerSetUp函数，除BasePrefetchDataLayer外的子类都应该实现自己的DataLayerSetUp函数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return true; }      // 数据并行时，数据读取层被多个solver共享
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;                         // 预处理参数
  shared_ptr<DataTransformer<Dtype> > data_transformer_;            // 图片预处理器
  bool output_labels_;                                              // 是否读取label 
};

// 预读取时的一组图片的blob和一组label的blob
template <typename Dtype>
class Batch {                           
 public:
  Blob<Dtype> data_, label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void InternalThreadEntry();                   // 用于预读取线程中的函数，不断的调用load_batch
  virtual void load_batch(Batch<Dtype>* batch) = 0;     // 实际的数据读取过程

  vector<shared_ptr<Batch<Dtype> > > prefetch_;         // 用于预读取的blob数组
  BlockingQueue<Batch<Dtype>*> prefetch_free_;          // 用于主线程和读取线程同步的阻塞队列，保存预读取blob的指针
  BlockingQueue<Batch<Dtype>*> prefetch_full_;
  Batch<Dtype>* prefetch_current_;                      // 当前正在网络中的预处理blob的指针

  Blob<Dtype> transformed_data_;                        // 通过set_cpu_data()将预处理后的数据保存到对应的blob中
};
```

# 实现细节
1. 在`BasePrefetchingDataLayer`的实现中，`Forward_cpu/gpu()`函数均使用了`set_cpu/gpu/data()`将输出blob的实际存储内存设置为预读取的blob的内存，这样就不用做数据拷贝。