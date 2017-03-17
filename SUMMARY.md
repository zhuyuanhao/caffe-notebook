# Summary

## 总体结构

* [概述](README.md)
* [代码结构](code_structure.md)

## 组件分析

* [Database](module/database.md)
* [SyncedMemory](module/syncedmem.md)
* [Blob](module/blob.md)
* [Layer](module/layer.md)
* [Layer工厂](module/layer_register.md)
* [Net](module/net.md)
* [Solver](module/solver.md)

## Layer分析

* [Layer概述](layers/readme.md)
* [数据读取层](layers/datalayer.md)
  * [DataTransformer](layers/datalayer/datatransformer.md)
  * [BaseDataLayer](layers/datalayer/basedatalayer.md)
  * [ImageDataLayer](layers/datalayer/imagedatalayer.md)
  * [DataLayer](layers/datalayer/datalayer.md)
* [CNN层](layers/cnnlayer.md)
  * [BaseConvolutionLayer](layers/cnnlayer/baseconvolutionlayer.md)
  * [Im2colLayer](layers/cnnlayer/im2collayer.md)
  * [CropLayer](layers/cnnlayer/croplayer.md)
  * [ConvolutionLayer](layers/cnnlayer/convolutionlayer.md)
  * [DeconvolutionLayer](layers/cnnlayer/deconvolutionlayer.md)
  * [PoolingLayer](layers/cnnlayer/poolinglayer.md)
  * [SPPLayer](layers/cnnlayer/spplayer.md)
* [RNN层](layers/rnnlayer.md)
  * [RecurrentLayer](layers/rnnlayer/recurrentlayer.md)
  * [RNNLayer](layers/rnnlayer/rnnlayer.md)
  * [LSTMLayer](layers/rnnlayer/lstmlayer.md)
* [通用层](layers/commonlayer.md)
  * [InnerProductLayer](layers/commonlayer/innerproductlayer.md)
  * [DropoutLayer](layers/commonlayer/dropoutlayer.md)
  * [EmbedLayer](layers/commonlayer/embedlayer.md)
* [正则化层](layers/normalizationlayer.md)
  * [LRNLayer](layers/normalizationlayer/lrnlayer.md)
  * [MVNLayer](layers/normalizationlayer/mvnlayer.md)
  * [BatchNormLayer](layers/normalizationlayer/batchnormlayer.md)
* [激活层](layers/activationlayer.md)
  * [ReLULayer](layers/activationlayer/relulayer.md)
  * [PReLULayer](layers/activationlayer/prelulayer.md)
  * [ELULayer](layers/activationlayer/elulayer.md)
  * [SigmoidLayer](layers/activationlayer/sigmoidlayer.md)
  * [TanHLayer](layers/activationlayer/tanhlayer.md)
  * [AbsValLayer](layers/activationlayer/absvallayer.md)
  * [PowerLayer](layers/activationlayer/powerlayer.md)
  * [ExpLayer](layers/activationlayer/explayer.md)
  * [LogLayer](layers/activationlayer/loglayer.md)
  * [BNLLLayer](layers/activationlayer/bnlllayer.md)
  * [ThresholdLayer](layers/activationlayer/thresholdlayer.md)
  * [BiasLayer](layers/normalizationlayer/biaslayer.md)
  * [ScaleLayer](layers/normalizationlayer/scalelayer.md)
* [工具层](layers/utilitylayer.md)
* [Loss层](layers/losslayer.md)
  * [LossLayer](layers/losslayer/losslayer.md)
  * [MultinomialLogisticLossLayer](layers/losslayer/multinomiallogisticlosslayer.md)
  * [InfogainLossLayer](layers/losslayer/infogainlosslayer.md)
  * [SoftmaxWithLossLayer](layers/losslayer/softmaxwithlosslayer.md)
  * [EuclideanLossLayer](layers/losslayer/euclideanlosslayer.md)
  * [HingeLossLayer](layers/losslayer/hingelosslayer.md)
  * [SigmoidCrossEntropyLossLayer](layers/losslayer/sigmoidcrossentropylosslayer.md)
  * [AccuracyLayer](layers/losslayer/accuracylayer.md)
  * [ContrastiveLossLayer](layers/losslayer/contrastivelosslayer.md)

## Solver分析

* [Solver概述](solvers/readme.md)

## 其他

* [相关工具使用](other_tools.md)
* [BLAS库](other/blas.md)
* [参考文档](reference.md)

