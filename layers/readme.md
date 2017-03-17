官方介绍：http://caffe.berkeleyvision.org/tutorial/layers.html

Caffe中的Layer主要分如下几类：
* 数据读取层
* CNN层
* RNN层
* 通用层
* 正则化层
* 激活函数层
* 工具层
* 损失层

大部分类都提供了CPU和GPU的实现，部分类还提供了CUDNN的包装，可以在模型中通过engine参数指定。

# 文件
```
# 数据读取层
# CNN层
# RNN层
# 通用层
# 正则化层
# 激活函数层
# 工具层
# 损失层


├── absval_layer.hpp
├── accuracy_layer.hpp
├── argmax_layer.hpp
├── batch_reindex_layer.hpp
├── bnll_layer.hpp
├── concat_layer.hpp
├── contrastive_loss_layer.hpp
├── cudnn_lcn_layer.hpp
├── cudnn_lrn_layer.hpp
├── cudnn_pooling_layer.hpp
├── cudnn_relu_layer.hpp
├── cudnn_sigmoid_layer.hpp
├── cudnn_softmax_layer.hpp
├── cudnn_tanh_layer.hpp
├── dummy_data_layer.hpp
├── eltwise_layer.hpp
├── elu_layer.hpp
├── euclidean_loss_layer.hpp
├── exp_layer.hpp
├── filter_layer.hpp
├── flatten_layer.hpp
├── hdf5_data_layer.hpp
├── hdf5_output_layer.hpp
├── hinge_loss_layer.hpp
├── infogain_loss_layer.hpp
├── input_layer.hpp
├── log_layer.hpp
├── loss_layer.hpp
├── memory_data_layer.hpp
├── multinomial_logistic_loss_layer.hpp
├── neuron_layer.hpp
├── parameter_layer.hpp
├── power_layer.hpp
├── prelu_layer.hpp
├── python_layer.hpp
├── reduction_layer.hpp
├── relu_layer.hpp
├── reshape_layer.hpp
├── sigmoid_cross_entropy_loss_layer.hpp
├── sigmoid_layer.hpp
├── silence_layer.hpp
├── slice_layer.hpp
├── softmax_layer.hpp
├── softmax_loss_layer.hpp
├── split_layer.hpp
├── tanh_layer.hpp
├── threshold_layer.hpp
├── tile_layer.hpp
└── window_data_layer.hpp
```
