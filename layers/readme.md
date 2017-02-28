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
base_data_layer.hpp
base_data_layer.cpp
base_data_layer.cu

image_data_layer.hpp
image_data_layer.cpp
data_layer.hpp
data_layer.cpp

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
├── base_conv_layer.hpp
├── 
├── batch_norm_layer.hpp
├── batch_reindex_layer.hpp
├── bias_layer.hpp
├── bnll_layer.hpp
├── concat_layer.hpp
├── contrastive_loss_layer.hpp
├── conv_layer.hpp
├── crop_layer.hpp
├── cudnn_conv_layer.hpp
├── cudnn_lcn_layer.hpp
├── cudnn_lrn_layer.hpp
├── cudnn_pooling_layer.hpp
├── cudnn_relu_layer.hpp
├── cudnn_sigmoid_layer.hpp
├── cudnn_softmax_layer.hpp
├── cudnn_tanh_layer.hpp
├── deconv_layer.hpp
├── dropout_layer.hpp
├── dummy_data_layer.hpp
├── eltwise_layer.hpp
├── elu_layer.hpp
├── embed_layer.hpp
├── euclidean_loss_layer.hpp
├── exp_layer.hpp
├── filter_layer.hpp
├── flatten_layer.hpp
├── hdf5_data_layer.hpp
├── hdf5_output_layer.hpp
├── hinge_loss_layer.hpp
├── im2col_layer.hpp
├── 
├── infogain_loss_layer.hpp
├── inner_product_layer.hpp
├── input_layer.hpp
├── log_layer.hpp
├── loss_layer.hpp
├── lrn_layer.hpp
├── lstm_layer.hpp
├── memory_data_layer.hpp
├── multinomial_logistic_loss_layer.hpp
├── mvn_layer.hpp
├── neuron_layer.hpp
├── parameter_layer.hpp
├── pooling_layer.hpp
├── power_layer.hpp
├── prelu_layer.hpp
├── python_layer.hpp
├── recurrent_layer.hpp
├── reduction_layer.hpp
├── relu_layer.hpp
├── reshape_layer.hpp
├── rnn_layer.hpp
├── scale_layer.hpp
├── sigmoid_cross_entropy_loss_layer.hpp
├── sigmoid_layer.hpp
├── silence_layer.hpp
├── slice_layer.hpp
├── softmax_layer.hpp
├── softmax_loss_layer.hpp
├── split_layer.hpp
├── spp_layer.hpp
├── tanh_layer.hpp
├── threshold_layer.hpp
├── tile_layer.hpp
└── window_data_layer.hpp

├── absval_layer.cpp
├── absval_layer.cu
├── accuracy_layer.cpp
├── argmax_layer.cpp
├── base_conv_layer.cpp

├── batch_norm_layer.cpp
├── batch_norm_layer.cu
├── batch_reindex_layer.cpp
├── batch_reindex_layer.cu
├── bias_layer.cpp
├── bias_layer.cu
├── bnll_layer.cpp
├── bnll_layer.cu
├── concat_layer.cpp
├── concat_layer.cu
├── contrastive_loss_layer.cpp
├── contrastive_loss_layer.cu
├── conv_layer.cpp
├── conv_layer.cu
├── crop_layer.cpp
├── crop_layer.cu
├── cudnn_conv_layer.cpp
├── cudnn_conv_layer.cu
├── cudnn_lcn_layer.cpp
├── cudnn_lcn_layer.cu
├── cudnn_lrn_layer.cpp
├── cudnn_lrn_layer.cu
├── cudnn_pooling_layer.cpp
├── cudnn_pooling_layer.cu
├── cudnn_relu_layer.cpp
├── cudnn_relu_layer.cu
├── cudnn_sigmoid_layer.cpp
├── cudnn_sigmoid_layer.cu
├── cudnn_softmax_layer.cpp
├── cudnn_softmax_layer.cu
├── cudnn_tanh_layer.cpp
├── cudnn_tanh_layer.cu
├── deconv_layer.cpp
├── deconv_layer.cu
├── dropout_layer.cpp
├── dropout_layer.cu
├── dummy_data_layer.cpp
├── eltwise_layer.cpp
├── eltwise_layer.cu
├── elu_layer.cpp
├── elu_layer.cu
├── embed_layer.cpp
├── embed_layer.cu
├── euclidean_loss_layer.cpp
├── euclidean_loss_layer.cu
├── exp_layer.cpp
├── exp_layer.cu
├── filter_layer.cpp
├── filter_layer.cu
├── flatten_layer.cpp
├── hdf5_data_layer.cpp
├── hdf5_data_layer.cu
├── hdf5_output_layer.cpp
├── hdf5_output_layer.cu
├── hinge_loss_layer.cpp
├── im2col_layer.cpp
├── im2col_layer.cu
├── 
├── infogain_loss_layer.cpp
├── inner_product_layer.cpp
├── inner_product_layer.cu
├── input_layer.cpp
├── log_layer.cpp
├── log_layer.cu
├── loss_layer.cpp
├── lrn_layer.cpp
├── lrn_layer.cu
├── lstm_layer.cpp
├── lstm_unit_layer.cpp
├── lstm_unit_layer.cu
├── memory_data_layer.cpp
├── multinomial_logistic_loss_layer.cpp
├── mvn_layer.cpp
├── mvn_layer.cu
├── neuron_layer.cpp
├── parameter_layer.cpp
├── pooling_layer.cpp
├── pooling_layer.cu
├── power_layer.cpp
├── power_layer.cu
├── prelu_layer.cpp
├── prelu_layer.cu
├── recurrent_layer.cpp
├── recurrent_layer.cu
├── reduction_layer.cpp
├── reduction_layer.cu
├── relu_layer.cpp
├── relu_layer.cu
├── reshape_layer.cpp
├── rnn_layer.cpp
├── scale_layer.cpp
├── scale_layer.cu
├── sigmoid_cross_entropy_loss_layer.cpp
├── sigmoid_cross_entropy_loss_layer.cu
├── sigmoid_layer.cpp
├── sigmoid_layer.cu
├── silence_layer.cpp
├── silence_layer.cu
├── slice_layer.cpp
├── slice_layer.cu
├── softmax_layer.cpp
├── softmax_layer.cu
├── softmax_loss_layer.cpp
├── softmax_loss_layer.cu
├── split_layer.cpp
├── split_layer.cu
├── spp_layer.cpp
├── tanh_layer.cpp
├── tanh_layer.cu
├── threshold_layer.cpp
├── threshold_layer.cu
├── tile_layer.cpp
├── tile_layer.cu
└── window_data_layer.cpp

```
