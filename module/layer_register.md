`layer`使用工厂模式来产生类的对象。类通过宏定义将自己在工厂的字典中注册，使用时使用类型名字符串做为key得到`layer`的实例对象。

对于某些`layer`类型，Caffe提供了自己的实现和CUDNN的实现，在编译时通过选项`USE_CUDNN := 1`选择。

# 文件
```
include/caffe/layer_factory.hpp
src/caffe/layer_factory.cpp
```

# 成员
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



