# 代码分析
## 总体结构
Caffe使用Protobuf文本格式（plaintext protocol buffer schema）定义Solver、Net、Layer。描述这些组件的文本格式的定义都放在caffe.proto文件中。



## 代码组件

### Layer


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



