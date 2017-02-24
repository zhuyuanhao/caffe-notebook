`layer`使用工厂模式来产生类的对象。类通过宏定义将自己在工厂的字典中注册，使用时使用类型名字符串做为key得到`layer`的实例对象。

对于某些`layer`类型，Caffe提供了自己的实现和CUDNN的实现，在运行时通过参数`engine`选择。

# 文件
```
include/caffe/layer_factory.hpp
src/caffe/layer_factory.cpp
```

# 对象
```cpp
// 注册工具类，不能被实例化，提供静态函数和字典变量完成Layer的注册和新建Layer
// 注册使用AddCreator函数，新建Layer使用CreateLayer函数
// 注册和新建时使用的Layer名时对应Layer的类名去掉最后的字符串Layer，比如class ReLULayer;使用"ReLU"作为注册名
template <typename Dtype>
class LayerRegistry {
 public:
  typedef shared_ptr<Layer<Dtype>> (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  // 静态函数，返回字符串到Layer Creator的字典，首次调用时会创建该字典
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();        // 静态变量，只会调用一次
    return *g_registry_;
  }
  // 静态函数，注册字典中的元素
  static void AddCreator(const string& type, Creator creator);
  
  // 静态函数，根据LayerParameter查询字典，并使用查询到的Creator创建对应的Layer并返回
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param);

  // 静态函数，查询注册字典中的所有Layer的注册名
  static vector<string> LayerTypeList();

 private:
  // 所有Layer注册工作由静态函数完成，所有不允许有LayerRegistry对象，故将构造函数设为私有
  LayerRegistry() {}

  // 静态函数，将注册字典中的所有Layer名作为一个字符串返回
  static string LayerTypeListString();
};

// 注册类，在构造函数中调用注册函数完成注册。被注册宏调用
template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};

// 注册宏，将type:creator的float和double类型注册
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

// 注册宏，新建一个Creator（这个Creator返回一个type##Layer的对象），然后将type:新建Creator的float和double注册
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe
```
```cpp
// 对CUDNN中有的Layer，以及Python Layer，在这里完成注册
// Creator函数在新建Layer时根据Layer名和engine参数选择合适的实现(CAFFE或CUDNN)

// 默认选择CAFFE的实现，某些参数CUDNN不支持的也选择CAFFE的实现
// 包括：Convolution, Pooling, LRN, ReLU, Sigmoid, Softmax, TanH
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetConvolutionLayer(
    const LayerParameter& param) {
  ConvolutionParameter conv_param = param.convolution_param();
  ConvolutionParameter_Engine engine = conv_param.engine();
#ifdef USE_CUDNN
  bool use_dilation = false;
  for (int i = 0; i < conv_param.dilation_size(); ++i) {
    if (conv_param.dilation(i) > 1) {
      use_dilation = true;
    }
  }
#endif
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (!use_dilation) {
      engine = ConvolutionParameter_Engine_CUDNN;
    }
#endif
  }
  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN) {
    if (use_dilation) {
      LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
                 << param.name();
    }
    return shared_ptr<Layer<Dtype> >(new CuDNNConvolutionLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);
```
# 实现细节
1. 在`include/caffe/layer_factory.hpp`中定义了注册类，注册时使用宏
```cpp
REGISTER_LAYER_CLASS(type)            # 使用默认的Creator
REGISTER_LAYER_CREATOR(type, creator)
```
使用时用静态函数
```cpp
shared_ptr<Layer<Dtype>> CreateLayer(const LayerParameter& param)
```
返回对应的一个`layer`对象

2. 在`src/caffe/layer_factory.cpp`注册了一些`layer`，使用特殊的`Creator`函数，使这些`layer`可以在运行时通过`engine`参数选择`CAFFE`或`CUDNN`实现的对象
3. 类型名一般使用对应Layer的类名去掉最后的字符串Layer，比如`class ReLULayer`使用`ReLU`作为注册的类名






