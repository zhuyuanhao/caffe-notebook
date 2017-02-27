`net`由Layer按照DAG方式组合而成，Layer之间通过Blob交换数据，还能通过sharedweights共享Blob。一个典型的网络以data layer开始，用于从存储设备载入图片或音频数据（以及label数据），以loss layer结束，用于计算分类或预测任务的损失值。

`net`的Forward过程又称为inference，Net::Forward从输入数据开始，经过每个Layer的Forward函数，最终在Loss Layer获得loss值。Backward过程又称为learning，Net::Backward从loss值开始，使用链式规则计算每个Layer的梯度值并保存。

初始化时，`net`会生成所有需要的Blobs和Layers并建立它们之间的连接关系，检查是否满足DAG约束。所有的数据计算和传递都是通过`net`中的相应函数控制。

`net`本身并不考虑是在CPU还是GPU上运算，通过`Caffe::mode()`和`Caffe::set_mode()`函数，各个Blobs和Layers自己选择是在CPU还是GPU上执行。

# 依赖
```protobuf
message NetState {                          // 网络状态类
  optional Phase phase = 1 [default = TEST];
  optional int32 level = 2 [default = 0];
  repeated string stage = 3;
}

message NetParameter {
  optional string name = 1;                         // 网络名

  repeated string input = 3;                        // 已废弃，原input
  repeated BlobShape input_shape = 8;               // 已废弃
  repeated int32 input_dim = 4;                     // 已废弃

  optional bool force_backward = 5 [default = false];// 是否强制网络中所有层都计算BP
  optional NetState state = 6;                      // 网络状态，包括phase，level，stage。用于某些layer的include/exclude选项
  optional bool debug_info = 7 [default = false];   // 是否在Forward，Backward，Update时输出debug信息

  repeated LayerParameter layer = 100;              // layer参数 
  repeated V1LayerParameter layers = 2;             // 已废弃
}
```

# 成员
```cpp
template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param);
  explicit Net(const string& param_file, Phase phase,
      const int level = 0, const vector<string>* stages = NULL);
  virtual ~Net() {}

  // 构造网络
  void Init(const NetParameter& param);

  // forward函数，按照layers_的顺序调用layer的Forward函数，并返回loss。同时调用所有回调函数before/after_forward_
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);

  // 按layers_的顺序并且考虑layer_need_backward_，调用layer的Backward函数，只更新梯度到diff中。同理会调用相关回调函数
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);
  
  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }
  
  // 将所有learnable_params_的diff都置为0
  void ClearParamDiffs();
  // 调用所有layer的Reshape()函数
  void Reshape();
  // 调用所有learable_params_的Update()函数
  void Update();
  // 根据param_owners_数组所述，将共享的param blob设置data_,diff_指向相同的SyncedMemory
  void ShareWeights();

  // 和另一个Net共享有相同layer_name的内部params的data_数据
  void ShareTrainedLayersWith(const Net* other);

  // net序列化相关函数
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  // 只保存name和所有layer的信息
  void ToProto(NetParameter* param, bool write_diff = false) const; 
  void ToHDF5(const string& filename, bool write_diff = false) const;

  // 查询内部数据的函数
  inline const string& name() const { return name_; }
  ...

  // 用于Init()初始化的函数，根据网络状态过滤一些layer
  static void FilterNet(const NetParameter& param, NetParameter* param_filtered);
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule, const string& layer_name);

  // 内部用于在forward/backward前后调用函数的包装类
  class Callback {
   protected:
    virtual void run(int layer) = 0;

    template <typename T>
    friend class Net;
  };

 protected:
  // 用于Init()的函数
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  // 用于显示调试信息的函数
  void ForwardDebugInfo(const int layer_id);
  void BackwardDebugInfo(const int layer_id);
  void UpdateDebugInfo(const int param_id);

  string name_;                                 // 网络名称 
  Phase phase_;                                 // 网络状态：TRAIN 或 TEST

  vector<shared_ptr<Layer<Dtype>>> layers_;     // 所有layer对象的数组
  vector<string> layer_names_;                  // layer名称数组
  map<string, int> layer_names_index_;          // layer字典：layer名称->下标
  vector<bool> layer_need_backward_;            // layer是否BP的数组

  // 每个blob至多只会被layer使用一次，net有分支时增加split layer产生多个blob
  vector<shared_ptr<Blob<Dtype>>> blobs_;       // 所有blob对象的数组
  vector<string> blob_names_;                   // blob名称数组
  map<string, int> blob_names_index_;           // blob字典：blob名称->下标
  vector<bool> blob_need_backward_;             // blob是否BP的数组

  // layer的输入是一个blob指针数组，输出也是一个blob指针数组
  vector<vector<Blob<Dtype>*>> bottom_vecs_;    // 所有layer的输入blob指针数组的数组
  vector<vector<int>> bottom_id_vecs_;          // layer的输入blob的下标
  vector<vector<bool>> bottom_need_backward_;   // layer的输入blob是否BP

  vector<vector<Blob<Dtype>*>> top_vecs_;       // 所有layer的输出blob指针数组的数组
  vector<vector<int>> top_id_vecs_;             // layer的输出blob的下标

  vector<Dtype> blob_loss_weights_;             // 所有数据blob的loss weight

  vector<vector<int>> param_id_vecs_;           // 每个layer的权值blob下标
  vector<int> param_owners_;                    // 记录当前param共享信息，-1非共享，否则记录共享的param下标
  vector<string> param_display_names_;          // 记录所有paramspec的名字，没有name的记录id下标
  vector<pair<int, int>> param_layer_indices_;  // layer下标和权值blob在layer内的序号
  map<string, int> param_names_index_;          // param名称和params_中下标的字典

  vector<int> net_input_blob_indices_;          // net的输入blob，即所有Input层的输出blob
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<int> net_output_blob_indices_;         // 所有未被layer使用的非自动blob作为net的输出blob
  vector<Blob<Dtype>*> net_output_blobs_;

  vector<shared_ptr<Blob<Dtype>>> params_;      // 所有权值blob，共享param分别存
  vector<Blob<Dtype>*> learnable_params_;       // 所有可学习的权值blob，共享param只存一份
  vector<int> learnable_param_ids_;             // 共享的param的下标相同，指向同一个learnable_params_元素
  vector<float> params_lr_;                     // 可学习权值blob的lr乘法因子
  vector<bool> has_params_lr_;                  // 可学习权值blob是否有lr乘法因子
  vector<float> params_weight_decay_;           // 可学习权值blob的衰减乘法因子
  vector<bool> has_params_decay_;               // 可学习权值blob是否有衰减乘法因子
  size_t memory_used_;                          // 所有输入输出blob的内存用量，不包括layer参数blob
  bool debug_info_;                             // 调试标志
  vector<Callback*> before_forward_;            // 一些回调函数
  vector<Callback*> after_forward_;
  vector<Callback*> before_backward_;
  vector<Callback*> after_backward_;

DISABLE_COPY_AND_ASSIGN(Net);
};
```

