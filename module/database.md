`Database`组件用于从LMDB/LevelDB数据库文件中读取数据，被`DataLayer`使用。

# 文件
```
include/caffe/util/db.hpp
include/caffe/util/db_lmdb.hpp
include/caffe/util/db_leveldb.hpp
```

# 依赖
1. `db.hpp`定义了数据库访问的接口基类，`db_lmdb.hpp`和`db_leveldb.hpp`分别定义了LMDB和LevelDB的继承类实现，他们分别依赖于系统的`lmdb.h`和`leveldb/db.h`、`leveldb/write_batch.h`头文件

# 对象
```cpp
enum Mode { READ, WRITE, NEW };         // 数据库访问模式

class Cursor {                          // 指针类，用于顺序读取数据
 public:
  Cursor() { }
  virtual ~Cursor() { }
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual string key() = 0;
  virtual string value() = 0;
  virtual bool valid() = 0;
  DISABLE_COPY_AND_ASSIGN(Cursor);
};

class Transaction {                     // 事务类，用于写数据
 public:
  Transaction() { }
  virtual ~Transaction() { }
  virtual void Put(const string& key, const string& value) = 0;
  virtual void Commit() = 0;
  DISABLE_COPY_AND_ASSIGN(Transaction);
};

class DB {                              // 数据库类
 public:
  DB() { }
  virtual ~DB() { }
  virtual void Open(const string& source, Mode mode) = 0;
  virtual void Close() = 0;
  virtual Cursor* NewCursor() = 0;
  virtual Transaction* NewTransaction() = 0;
  DISABLE_COPY_AND_ASSIGN(DB);
};

DB* GetDB(DataParameter::DB backend);   // 可以通过Proto定义的枚举或字符串确定实际使用的数据库类型
DB* GetDB(const string& backend);
```