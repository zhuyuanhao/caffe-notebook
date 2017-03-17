`embed`的作用主要在于学习词语的distributed representation并将极其稀疏的one-hot编码的词语进行降维。`EmbedLayer`的作用相当于在`InnerProductLayer`之前先进行`embed`操作。

# 文件
```
include/caffe/layers/embed_layer.hpp
src/caffe/layers/embed_layer.cpp
src/caffe/layers/embed_layer.cu
```

# 原理

# 参数
```protobuf
message LayerParameter {
  optional EmbedParameter embed_param = 137;
}
  
message EmbedParameter {
  optional uint32 num_output = 1; 
  optional uint32 input_dim = 2;

  optional bool bias_term = 3 [default = true]; 
  optional FillerParameter weight_filler = 4; 
  optional FillerParameter bias_filler = 5; 

}
```