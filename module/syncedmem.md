`SyncedMemory`类可以看作一个`void*`类型的数组，这个数组可以在CPU和GPU的代码中访问，数组数据在CPU和GPU之间按需同步。
# 文件
```
include/caffe/syncedmem.hpp
src/caffe/syncedmem.cpp
```
# 依赖
CPU中的内存可以使用`malloc`分配普通内存，也可以使用`cudaMallocHost`分配cuda pinned cpu memory。
在`include/caffe/syncedmem.hpp`中定义了一组分配/释放函数，当当前环境是是`Caffe::mode() == Caffe::GPU`时，将使用cuda pinned memory，这样避免了dynamic pinning for transfers(DMA)，在多卡时速度更快更稳定。
```cpp
void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda);
void CaffeFreeHost(void* ptr, bool use_cuda)
```

# 成员
```cpp
/*
SyncedMemory类可以看作一个void*类型的数组，这个数组可以在CPU和GPU的代码中访问，数据在CPU和GPU之间按需同步。
内存分配策略：
1. 初始化时不分配内存，只记录大小，head_状态为UNINITIALIZED，表示CPU和GPU中都没有分配内存
2. 调用了6个函数之一[set/mutable_]cpu/gpu_data()后才会分配内存，并将head_修改为对应的位置，注意调用set/mutable_*后head_一定不为SYNCED
3. cpu_data(),gpu_data()按需同步数据，并返回内部cpu_ptr_,gpu_ptr_的常量指针
4. mutable_cpu_data(),mutable_gpu_data()按需同步数据，并返回内部变量的普通指针
5. set_cpu_data(void*),set_gpu_data(void*)会释放内部指针的内存，然后置为传入的指针指向的内存
6. to_cpu(),to_gpu()是私有函数，检查并执行实际的数据同步操作，会被[mutable_]cpu/gpu_data()调用
7. head_为SYNCED表示CPU和GPU中的数据一样新，head_为HEAD_AT_CPU标示CPU中数据比GPU新或GPU中还没有分配内存，
head_为HEAD_AT_GPU时同理
*/
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);  // 在stream中使用将CPU数据异步方式同步到GPU，在data prefetch时使用
#endif

 private:
  void check_device();          // 检查当前线程对应的GPU ID是否和新建时保存在device_中的GPU ID一致

  void to_cpu();                // 按需同步数据到CPU
  void to_gpu();                // 按需同步数据到GPU
  void* cpu_ptr_;               // CPU中内存块的首地址
  void* gpu_ptr_;               // GPU中内存块的首地址
  size_t size_;                 // CPU&GPU中内存的字节数
  SyncedHead head_;             // 当前最新的数据的位置
  bool own_cpu_data_;           // cpu_ptr_指向的内存是否是自己分配的，若是，不用后要负责释放
  bool cpu_malloc_use_cuda_;    // cpu_ptr_指向的内存是否是pinned memory
  bool own_gpu_data_;           // gpu_ptr_的内存是否是自己分配的，若是，要负责释放
  int device_;                  // 记录新建内存时的GPU ID

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);    // 禁用拷贝构造和赋值运算符
};  // class SyncedMemory

```