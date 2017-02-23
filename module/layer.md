`layer`可以看作一个函数，读取0个或多个`blob`的数据作为输入，和内部保存的数据（也使用`blob`格式存储）进行计算后将结果输出到0或多个`blob`。

# 文件
```
include/caffe/layer.hpp
src/caffe/layer.cpp
include/caffe/layer_factory.hpp
src/caffe/layer_factory.cpp
```
# 依赖
1. `layer`中数据均采用`Blob`类保存，包括输入、输出、内部参数
2. `layer`的初始化参数保存在`src/caffe/proto/caffe.proto`定义的`LayerParameter`类中，在程序启动后从`*.proto`模型描述文件中获得

```protobuf
enum Phase {
   TRAIN = 0;
   TEST = 1;
}

message NetState {                          // 网络状态类
  optional Phase phase = 1 [default = TEST];
  optional int32 level = 2 [default = 0];
  repeated string stage = 3;
}

message NetStateRule {                      // 网络状态类规则类，用于判断当前网络状态是否满足网络状态规则
  optional Phase phase = 1;
  optional int32 min_level = 2;
  optional int32 max_level = 3;
  repeated string stage = 4;
  repeated string not_stage = 5;
}

message ParamSpec {                         // 训练参数类
  optional string name = 1;                 // 在layer间共享训练参数时使用，其他情况下留空
  optional DimCheckMode share_mode = 2;     // layer间共享时是否需要对应的blob内部参数的维度一致
  enum DimCheckMode {
    STRICT = 0;                             // 默认值，需要blob的所有维度一致
    PERMISSIVE = 1;                         // 只需要blob的元素总数(count)一致
  }
  optional float lr_mult = 3 [default = 1.0];   // lr的乘法因子
  optional float decay_mult = 4 [default = 1.0];// weight decay的乘法因子
}

message LayerParameter {
  optional string name = 1;                 // layer名称
  optional string type = 2;                 // layer类型，用于在layer_factory中查找
  repeated string bottom = 3;               // 输入blob的名称数组
  repeated string top = 4;                  // 输出blob的名称数组
  optional Phase phase = 10;                // layer状态：TRAIN或TEST
  repeated float loss_weight = 5;           // 为每个输出blob设置的loss weight，通常为0或1
  repeated ParamSpec param = 6;             // 训练参数类数组
  repeated BlobProto blobs = 7;             // 保存的layer内部数据数组
  repeated bool propagate_down = 11;        // BP时是否强制计算输入blob的梯度（diff）的数组，长度为0或与输入blob的个数一致
  repeated NetStateRule include = 8;        // 根据当前NetState判断是否包含当前layer的配置数组
  repeated NetStateRule exclude = 9;
  optional TransformationParameter transform_param = 100;   // 数据读取层的预处理参数类
  optional LossParameter loss_param = 101;                  // 所有类型的loss layer共有的参数
  optional AccuracyParameter accuracy_param = 102;      // 每个layer自己特有的参数类
  optional ...
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
template <typename Dtype>
class Layer {
 public:
  // 按照设计，所有继承的Layer不要有自己的构造函数。初始化工作都放在SetUp()中。
  explicit Layer(const LayerParameter& param);
  virtual ~Layer() {}

  // 初始化函数，注意它不是虚函数，net初始化时会按层调用它们的SetUp函数
  void SetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);       // 虚函数，检查bottom和top数组的元素数量是否满足要求
    LayerSetUp(bottom, top);            // 虚函数，特定的Layer的初始化工作
    Reshape(bottom, top);               // 虚函数，根据bottom对内部的和top中的blob作reshape
    SetLossWeights(top);                // 设置top中每个blob的loss weight
  }

  // Forward函数有三步：Reshape，Forward_cpu/gpu，计算并返回loss
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  // Backward只调用Backward_cpu/gpu，其中propagate_down表示对应的bottom的diff是否计算
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual inline const char* type() const;  // 虚函数，返回Layer的类型字符串，对应于LayerParameter类中的type字段

  // 将内部layer_param_和blobs_的数据写入param
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  // 获取内部layer_param_和blobs_数据
  const LayerParameter& layer_param() const;
  vector<shared_ptr<Blob<Dtype> > >& blobs();

  // 获取或设置某个top blob的loss weight值
  inline Dtype loss(const int top_index) const;
  inline void set_loss(const int top_index, const Dtype value);

  // 查询或设置param_propagate_down_,表示内部的blobs_参数是否需要计算diff
  inline bool param_propagate_down(const int param_id);
  inline void set_param_propagate_down(const int param_id, const bool value);

  // 虚函数，特定Layer的初始化函数，一般根据参数初始化内部变量
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  // 虚函数，根据bottom blobs的大小，调整top blobs和内部blobs的大小
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) = 0;

  // 检测bottom blob和top blob的个数是否满足要求的函数
  // 都是虚函数，每个Layer根据自己的要求覆盖某些函数，默认情况下都返回-1，表示不检查该项
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return -1; }
  virtual inline int MaxBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return -1; }
  virtual inline int MaxTopBlobs() const { return -1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }
  
  // 虚函数，是否自动产生需要数目的top blobs，在Net::Init中调用
  virtual inline bool AutoTopBlobs() const;
  // 虚函数，是否某个bottom blob允许强制计算Backward
  virtual inline bool AllowForceBackward(const int bottom_index) const;

 protected:
  LayerParameter layer_param_;                      // Layer的参数
  Phase phase_;                                     // Layer的状态，TRAIN或者TEST
  vector<shared_ptr<Blob<Dtype> > > blobs_;         // Layer的内部参数数组，是需要学习的参数
  vector<bool> param_propagate_down_;               // 判断内部的blobs_参数是否需要计算diff的数组
  vector<Dtype> loss_;                              // 存储每个top blob的loss weight，为0表示不计算loss

  // 四个虚函数，Forward和Backward时，在CPU或GPU计算的函数
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // 虚函数，SetUp时检查bottom和top的blob个数是否满足要求，它调用的函数也都是虚函数
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  // 根据layer_param_中的loss_weight数组，设置每个top的loss weight
  inline void SetLossWeights(const vector<Blob<Dtype>*>& top);
  
 private:
  DISABLE_COPY_AND_ASSIGN(Layer);
};
```
// 检查7项，包括{ExactNum,Min,Max} {Bottom,Top}和Bottom Equal Top 
// loss weight不仅保存在Layer的loss_数组中，对应top blob的diff的所有元素都赋值为loss weight