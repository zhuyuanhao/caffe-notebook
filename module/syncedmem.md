`SyncedMemory`类可以看作一个`void*`类型的数组，这个数组可以在CPU和GPU的代码中访问，数据在CPU和GPU之间按需同步。

# 文件
```
include/caffe/syncedmem.hpp
src/caffe/syncedmem.cpp
```
# 依赖
CPU中的内存可以使用`malloc`分配普通内存，也可以使用`cudaMallocHost`分配cuda pinned cpu memory。
在`include/caffe/syncedmem.hpp`中定义了一组函数，当当前环境是是`Caffe::mode() == Caffe::GPU`时，将使用cuda pinned memory。
```cpp
void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda);
void CaffeFreeHost(void* ptr, bool use_cuda)
```

# 成员
```cpp
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
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory
```
