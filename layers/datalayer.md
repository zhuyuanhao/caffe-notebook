数据读取层负责从存储介质中读取训练数据，Caffe通过多线程预取的方式使读取数据和数据计算分别用不同的线程，提升CPU使用效率。

# 数据类型
caffe输入数据可以是`数据库lmdb(default)/leveldb、内存文件、磁盘文件hdf5/.mat/jpg/png`等格式。
caffe了提供`tools/convert_imageset.cpp`文件。编译之后，生成对应的可执行文件放在 `build/tools/` 下面，这个文件的作用就是用于将图片文件转换成caffe框架中能直接使用的db文件。
caffe还提供了一个计算均值的文件`tools/compute_image_mean.cpp`，计算图片集的均值，保存在`.binaryproto`文件中供读取时做data_transform使用。



# 数据转换

