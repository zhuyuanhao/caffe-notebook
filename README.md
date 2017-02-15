# 代码分析
## 总体结构
Net由Layer组成，Layer之间可以是任意的DAG关系。Layer之间通过Blob做数据传递，数据从Layer的bottom blob(s)传输到top blob(s)。

Caffe使用Protobuf文本格式（plaintext protocol buffer schema）定义Solver、Net、Layer。描述这些组件的文本格式的定义都放在caffe.proto文件中。

## 输入数据
caffe输入数据可以是数据库（lmdb(default)，leveldb），内存文件，磁盘文件（hdf5/.mat/图片格式）等格式。caffe为我们提供tools/convert_imageset.cpp文件。编译之后，生成对应的可执行文件放在 buile/tools/ 下面，这个文件的作用就是用于将图片文件转换成caffe框架中能直接使用的db文件。
caffe还提供了一个计算均值的文件tools/compute_image_mean.cpp，计算图片集的均值，保存在.binaryproto文件中供读取时做data_transform使用。
lmdb和leveldb
- 它们都是键/值对（Key/Value Pair）嵌入式数据库管理系统编程库。
- 虽然lmdb的内存消耗是leveldb的1.1倍，但是lmdb的速度比leveldb快10%至15%，更重要的是lmdb允许多种训练模型同时读取同一组数据集。
- 因此lmdb取代了leveldb成为Caffe默认的数据集生成格式。

## 代码组件
### Blob
Blob提供统一的数据抽象。数据都按照N维数组存储，使用C的连续格式（行优先），内存空间惰性分配，CPU和GPU之间的数据按需同步。
一般格式：
- 对于数据：(Batch)Number*Channel*Height*Width
- 对于卷积权重：Output*Input*Height*Width
- 对于卷积偏置：Output*1*1*1

实现细节：
- reshape函数申明数据大小。（惰性分配？）
- blob包含两类数据：data和diff。diff存储网络计算出来的梯度(gradient)
- 获取不可修改数据：const Dtype* cpu_data() const; 获取可修改数据：Dtype* mutable_cpu_data();
- 使用SyncedMem类同步CPU和GPU的数据，每次调用mutable_*函数后再切换到另一个设备都会涉及数据拷贝操作。所以尽量使用非mutable_*函数获取数据，并且不保存数据指针(防止修改)。

### Layer
主要包含setup，forward，backward函数。Forward和Backward函数都包含CPU和GPU版本，可以只实现CPU版本（用于快速验证），此时数据会在GPU和CPU间做两次拷贝。
主要类型：
http://caffe.berkeleyvision.org/tutorial/layers.html
https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
Vision Layer：Convolution，Pooling，LocalResponseNormalization，
Activation Layer：ReLU，Sigmoid，BNLL
Data Layer：Data，MemoryData，HDF5Data
Common Layer：InnerProduct，Reshape
相应的头文件为：
- layer.hpp: 父类Layer，定义所有layer的基本接口。
- data_layers.hpp: 继承自父类Layer，定义与输入数据操作相关的子Layer，例如DataLayer，HDF5DataLayer和ImageDataLayer等。
- vision_layers.hpp: 继承自父类Layer，定义与特征表达相关的子Layer，例如ConvolutionLayer，PoolingLayer和LRNLayer等。
- neuron_layers.hpp: 继承自父类Layer，定义与非线性变换相关的子Layer，例如ReLULayer，TanHLayer和SigmoidLayer等。
- loss_layers.hpp: 继承自父类Layer，定义与输出误差计算相关的子Layer，例如EuclideanLossLayer，SoftmaxWithLossLayer和HingeLossLayer等。
- common_layers.hpp: 继承自父类Layer，定义与中间结果数据变形、逐元素操作相关的子Layer，例如ConcatLayer，InnerProductLayer和SoftmaxLayer等。
- layer_factory.hpp: Layer工厂模式类，负责维护现有可用layer和相应layer构造方法的映射表。

Loss (Layer)
计算损失函数，决定了最终模型的类型，分类、预测等。
主要类型：Softmax，Euclidean，Hinge
http://caffe.berkeleyvision.org/tutorial/layers.html

Data (Layer)
数据输入层一般只有Top的Blobs，包括data blob和label blob。
数据会被预处理，并且使用prefetch优化，即计算当前数据时，预先读取下一个batch的数据。

### Net
由Layer按照DGA方式组合而成，Layer之间通过Blob交换数据，还能通过sharedweights共享Blob。一个典型的网络以data layer开始，用于从存储设备载入图片或音频数据（以及label数据），以loss layer结束，用于计算分类或预测任务的损失值。Net的Forward过程又称为inference，Net::Forward从输入数据开始，经过每个Layer的Forward函数，最终在Loss Layer获得loss值。Backward过程又称为learning，Net::Backward从loss值开始，使用链式规则计算每个Layer的梯度值并保存。
初始化时，Net会生成所有需要的Blobs和Layers并建立它们之间的连接关系，检查是否满足DAG约束。所有的数据计算和传递都是通过Net中的相应函数控制。
Net本身并不考虑是在CPU还是GPU上运算，通过Caffe::mode()和Caffe::set_mode()函数，各个Blobs和Layers自己选择是在CPU还是GPU上执行。

### Solver
Solver负责网络参数的更新，通过使用不同的规则控制梯度（gradients）更新到参数（parameter）中的方式。
主要类型：SGD，AdaDelta，AdaGrad，Adam，Nesterov，RMSProp
http://caffe.berkeleyvision.org/tutorial/solver.html
参数：
base_lr表示基础学习率，lr_policy用于控制学习率在每次迭代中的调整，可以设置为下面这些值，相应的学习率的计算为：
  - - fixed:　　 保持base_lr不变.
  - - step: 　　 如果设置为step,则还需要设置一个stepsize, 返回 base_lr * gamma ^ (floor(iter / stepsize)),其中iter表示当前的迭代次数
  - - exp: 　　返回base_lr * gamma ^ iter， iter为当前迭代次数
  - - inv:　　 如果设置为inv,还需要设置一个power, 返回base_lr * (1 + gamma * iter) ^ (- power)
  - - multistep: 如果设置为multistep,则还需要设置一个stepvalue。这个参数和step很相似，step是均匀等间隔变化，而multistep则是根据 stepvalue值变化
  - - poly: 　　 学习率进行多项式误差, 返回 base_lr (1 - iter/max_iter) ^ (power)
  - - sigmoid:　学习率进行sigmod衰减，返回 base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))

## 操作接口
http://caffe.berkeleyvision.org/tutorial/interfaces.html
### 命令行操作
```bash
Training
# train LeNet
caffe train -solver examples/mnist/lenet_solver.prototxt
# train on GPU 2
caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 2
# train on GPUs 0 & 1 (doubling the batch size)
caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 0,1
# train on all GPUs (multiplying batch size by number of devices)
caffe train -solver examples/mnist/lenet_solver.prototxt -gpu all
# resume training from the half-way point snapshot
caffe train -solver examples/mnist/lenet_solver.prototxt -snapshot examples/mnist/lenet_iter_5000.solverstate
# fine-tune CaffeNet model weights for style recognition
caffe train -solver examples/finetuning_on_flickr_style/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
```
Testing
```
# score the learned LeNet model on the validation set as defined in the
# model architeture lenet_train_test.prototxt
caffe test -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 100
```
Benchmarking
```
# (These example calls require you complete the LeNet / MNIST example first.)
# time LeNet training on CPU for 10 iterations
caffe time -model examples/mnist/lenet_train_test.prototxt -iterations 10
# time LeNet training on GPU for the default 50 iterations
caffe time -model examples/mnist/lenet_train_test.prototxt -gpu 0
# time a model architecture with the given weights on the first GPU for 10 iterations
caffe time -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 10
```
Diagnostics
```
# query the first device
caffe device_query -gpu 0
```

# 性能
K40（ECC off减少内存占用，boost clock enabled加快始终频率，cuDNN）
Traning：19.2s/20iter 5120 imgs
Testing：60.7s 50k imgs

http://caffe.berkeleyvision.org/performance_hardware.html

# 安装
http://caffe.berkeleyvision.org/installation.html

# 课程
http://vision.stanford.edu/teaching/cs231n/index.html
http://neuralnetworksanddeeplearning.com/



