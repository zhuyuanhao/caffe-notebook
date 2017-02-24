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

