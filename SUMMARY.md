# Summary

## 总体结构

* [概述](README.md)
* [代码结构](code_structure.md)

## 组件分析

* [Database](module/database.md)
* [SyncedMemory](module/syncedmem.md)
* [Blob](module/blob.md)
* [Layer](module/layer.md)
* [Layer注册机制](module/layer_register.md)
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
* [通用层](layers/commonlayer.md)
* [正则化层](layers/normalizationlayer.md)
* [激活函数层](layers/activationlayer.md)
* [工具层](layers/utilitylayer.md)
* [损失层](layers/losslayer.md)

## Solver分析

* [Solver概述](solvers/readme.md)

## 其他

* [相关工具使用](other_tools.md)
* [BLAS库](other/blas.md)
* [参考文档](reference.md)

