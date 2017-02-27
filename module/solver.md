Solver负责网络参数的更新，通过使用不同的规则控制梯度（gradients）更新到参数（parameter）中的方式。
主要类型：SGD，AdaDelta，AdaGrad，Adam，Nesterov，RMSProp
http://caffe.berkeleyvision.org/tutorial/solver.html
参数：
base_lr表示基础学习率，lr_policy用于控制学习率在每次迭代中的调整，可以设置为下面这些值，相应的学习率的计算为：
  - fixed:　　 保持base_lr不变.
  - step: 　　 如果设置为step,则还需要设置一个stepsize, 返回 base_lr * gamma ^ (floor(iter / stepsize)),其中iter表示当前的迭代次数
  - exp: 　　返回base_lr * gamma ^ iter， iter为当前迭代次数
  - inv:　　 如果设置为inv,还需要设置一个power, 返回base_lr * (1 + gamma * iter) ^ (- power)
  - multistep: 如果设置为multistep,则还需要设置一个stepvalue。这个参数和step很相似，step是均匀等间隔变化，而multistep则是根据 stepvalue值变化
  - poly: 　　 学习率进行多项式误差, 返回 base_lr (1 - iter/max_iter) ^ (power)
  - sigmoid:　学习率进行sigmod衰减，返回 base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
  
# 文件
```
include/caffe/solver.hpp
src/caffe/solver.cpp
include/caffe/solver_factory.hpp
src/caffe/solver_factory.cpp
include/caffe/sgd_solvers.hpp
src/caffe/solvers/*.cpp[cu]
```

# 依赖
```protobuf
// 保存snapshot时记录solver状态
message SolverState {
  optional int32 iter = 1;                          // 当前iter
  optional string learned_net = 2;                  // 存储网络的文件名称
  repeated BlobProto history = 3;                   // solver的历史
  optional int32 current_step = 4 [default = 0];    // 学习率的当前step
}

message SolverParameter {
  // train和test网络的定义
  optional string net = 24;
  optional NetParameter net_param = 25;
  optional string train_net = 1;
  repeated string test_net = 2;
  optional NetParameter train_net_param = 21;
  repeated NetParameter test_net_param = 22;

  // train和test网络的状态参数
  optional NetState train_state = 26;
  repeated NetState test_state = 27;

  // test网络参数
  repeated int32 test_iter = 3;                             // 每个测试网络的每次的迭代次数
  optional int32 test_interval = 4 [default = 0];
  optional bool test_compute_loss = 19 [default = false];
  optional bool test_initialization = 32 [default = true];  // train之前先作test，可用于保证内存容量和计算初始loss
  // train参数
  optional float base_lr = 5;
  optional int32 display = 6;
  optional int32 average_loss = 33 [default = 1];
  optional int32 max_iter = 7;
  optional int32 iter_size = 36 [default = 1];              // 每个iter对同一组数据计算多次后去平均loss

  // 网络update参数时的参数
  // 学习率衰减类型，包含
  //    - fixed: return base_lr
  //    - step: return base_lr * gamma ^ (floor(iter / step))
  //    - exp: return base_lr * gamma ^ iter
  //    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
  //    - multistep: similar to step but it allows non uniform steps defined by stepvalue
  //    - poly: the effective learning rate follows a polynomial decay, to be
  //      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
  //    - sigmoid: the effective learning rate follows a sigmod decay
  //      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
  optional string lr_policy = 8;
  optional float gamma = 9;
  optional float power = 10;
  optional float momentum = 11;
  optional float weight_decay = 12;
  optional string regularization_type = 29 [default = "L2"];        // 正则类型，L1或L2
  optional int32 stepsize = 13;                                     // step衰减规则的参数
  repeated int32 stepvalue = 34;                                    // multistep规则的参数

  // Set clip_gradients to >= 0 to clip parameter gradients to that L2 norm,
  // whenever their actual L2 norm is larger.
  optional float clip_gradients = 35 [default = -1];

  // snapshot参数
  optional int32 snapshot = 14 [default = 0];
  optional string snapshot_prefix = 15;
  optional bool snapshot_diff = 16 [default = false];
  enum SnapshotFormat {
    HDF5 = 0;
    BINARYPROTO = 1;
  }
  optional SnapshotFormat snapshot_format = 37 [default = BINARYPROTO];

  // solver模式参数
  enum SolverMode {
    CPU = 0;
    GPU = 1;
  }
  optional SolverMode solver_mode = 17 [default = GPU];
  optional int32 device_id = 18 [default = 0];
  optional int64 random_seed = 20 [default = -1];
  optional string type = 40 [default = "SGD"];
  optional bool debug_info = 23 [default = false];
  optional bool snapshot_after_train = 28 [default = true];
  // 是否每层作reduce，用于数据并行时同时计算和通信
  optional bool layer_wise_reduce = 41 [default = true];

  // RMSProp, AdaGrad and AdaDelta and Adam的参数
  optional float delta = 31 [default = 1e-8];
  // Adam solver的参数
  optional float momentum2 = 39 [default = 0.999];
  // RMSProp的参数
  // MeanSquare(t) = rms_decay*MeanSquare(t-1) + (1-rms_decay)*SquareGradient(t)
  optional float rms_decay = 38 [default = 0.99];
  // 已废弃，现在用字段string type = 40
  enum SolverType {
    SGD = 0;
    NESTEROV = 1;
    ADAGRAD = 2;
    RMSPROP = 3;
    ADADELTA = 4;
    ADAM = 5;
  }
  // 已废弃
  optional SolverType solver_type = 30 [default = SGD];
}
```

# 成员
```cpp
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);

  // 初始化solver，会初始化train和test net
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();

  // 事件响应相关函数
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();

  // 训练函数
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  void Restore(const char* resume_file);

  void Snapshot();
  virtual ~Solver() {}
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() const { return iter_; }

  // 定义训练时的相关回调函数
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  void CheckSnapshotWritePermissions();
  virtual inline const char* type() const { return ""; }

 protected:
  // 虚函数，根据diff和当前状态更新data
  virtual void ApplyUpdate() = 0;
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();

  void TestAll();
  void Test(const int test_net_id = 0);

  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;

  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_;                       // solver参数
  int iter_;                                    // 当前train的迭代次数
  int current_step_;                            //
  shared_ptr<Net<Dtype> > net_;                 // train net，只能有一个。数据并行时，多个train net是一样的
  vector<shared_ptr<Net<Dtype> > > test_nets_;  // test net数组，可以有多个不同的test net，但都在root solver中
  vector<Callback*> callbacks_;                 // 保存回调函数
  vector<Dtype> losses_;                        // 记录若干次迭代的loss，用于平滑loss
  Dtype smoothed_loss_;                         // 平滑后的loss值

  ActionCallback action_request_function_;      // 键盘事件的响应函数
  bool requested_early_exit_;                   // 内部循环中判断这个标志来决定是否退出当前循环

  Timer iteration_timer_;                       // 训练计时器
  float iterations_last_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};
```
