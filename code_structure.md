## 代码结构
```bash
caffe.cloc                // 配置：代码统计工具cloc
.Doxyfile                 // 配置：文档生成工具doxygen
.travis.yml               // 配置：持续集成工具TravisCI
cmake/                    // 配置：cmake
CMakeLists.txt            // 配置：cmake
Makefile                  // 配置：make
Makefile.config           // 配置：make
Makefile.config.example   // 配置：make
build -> .build_release/  // 编译：编译结果存放位置
distribute/               // 编译：生成发布包的位置，用于迁移
models/                   // 文档：示例模型alexnet,googlenet,rcnn
docs/                     // 文档：caffe官方文档
examples/                 // 文档：caffe使用手册（ipython）
CONTRIBUTING.md           // 文档：贡献代码
CONTRIBUTORS.md           // 文档：贡献者
INSTALL.md                // 文档：安装说明
README.md                 // 文档：项目说明
LICENSE                   // 文档：许可证
data/                     // 脚本：下载数据集mnist,cifar,imagenet
scripts/                  // 脚本：生成本地文档，下载已训练模型
docker/                   // 脚本：生成docker
matlab/                   // 代码：matlab接口
python/                   // 代码：python接口
include/                  // 核心代码：所有头文件
src/                      // 核心代码：所有实现文件
tools/                    // 核心代码：最终执行文件和其他工具
```
## include结构
```
include/
└── caffe
    ├── blob.hpp
    ├── caffe.hpp
    ├── common.hpp
    ├── data_transformer.hpp
    ├── filler.hpp
    ├── internal_thread.hpp
    ├── layer_factory.hpp
    ├── layer.hpp
    ├── layers\
    ├── net.hpp
    ├── parallel.hpp
    ├── sgd_solvers.hpp
    ├── solver_factory.hpp
    ├── solver.hpp
    ├── syncedmem.hpp
    ├── test
    │   ├── test_caffe_main.hpp
    │   └── test_gradient_check_util.hpp
    └── util
        ├── benchmark.hpp
        ├── blocking_queue.hpp
        ├── cudnn.hpp
        ├── db.hpp
        ├── db_leveldb.hpp
        ├── db_lmdb.hpp
        ├── device_alternate.hpp
        ├── format.hpp
        ├── gpu_util.cuh
        ├── hdf5.hpp
        ├── im2col.hpp
        ├── insert_splits.hpp
        ├── io.hpp
        ├── math_functions.hpp
        ├── mkl_alternate.hpp
        ├── nccl.hpp
        ├── rng.hpp
        ├── signal_handler.h
        └── upgrade_proto.hpp
```
## src结构
```
src
├── caffe
│   ├── blob.cpp
│   ├── CMakeLists.txt
│   ├── common.cpp
│   ├── data_transformer.cpp
│   ├── internal_thread.cpp
│   ├── layer.cpp
│   ├── layer_factory.cpp
│   ├── layers\
│   ├── net.cpp
│   ├── parallel.cpp
│   ├── proto
│   │   └── caffe.proto
│   ├── solver.cpp
│   ├── solvers\
│   ├── syncedmem.cpp
│   ├── test\
│   └── util
│       ├── benchmark.cpp
│       ├── blocking_queue.cpp
│       ├── cudnn.cpp
│       ├── db.cpp
│       ├── db_leveldb.cpp
│       ├── db_lmdb.cpp
│       ├── hdf5.cpp
│       ├── im2col.cpp
│       ├── im2col.cu
│       ├── insert_splits.cpp
│       ├── io.cpp
│       ├── math_functions.cpp
│       ├── math_functions.cu
│       ├── signal_handler.cpp
│       └── upgrade_proto.cpp
└── gtest
    ├── CMakeLists.txt
    ├── gtest-all.cpp
    ├── gtest.h
    └── gtest_main.cc
```
## tools结构
```
tools/
├── caffe.cpp
├── CMakeLists.txt
├── compute_image_mean.cpp
├── convert_imageset.cpp
├── extra
│   ├── extract_seconds.py
│   ├── launch_resize_and_crop_images.sh
│   ├── parse_log.py
│   ├── parse_log.sh
│   ├── plot_log.gnuplot.example
│   ├── plot_training_log.py.example
│   ├── resize_and_crop_images.py
│   └── summarize.py
├── extract_features.cpp
├── upgrade_net_proto_binary.cpp
├── upgrade_net_proto_text.cpp
└── upgrade_solver_proto_text.cpp
```