`SyncedMemory`类可以看作一个`void*`类型的数组，这个数组内部提供了CPU和GPU间的数据按需同步。

# 文件
```
include/caffe/syncedmem.hpp
src/caffe/syncedmem.cpp
```
# 依赖
```cpp
/* 通过Caffe::mode()函数判断当前模式

*/
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
