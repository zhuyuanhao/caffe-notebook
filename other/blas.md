# BLAS概述
Level 1
Vector operations, e.g. $$y = \alpha x + y$$  向量操作
Level 2
Matrix-vector operations, e.g. $$y = \alpha A x + \beta y$$  矩阵与向量操作
Level 3
Matrix-matrix operations, e.g. $$C = \alpha A B + C$$    矩阵与矩阵的操作

## 函数名规则
### 操作
DOT
scalar product, x^T y
AXPY
vector sum, /alpha x + y
MV
matrix-vector product, A x
SV
matrix-vector solve, inv(A) x
MM
matrix-matrix product, A B
SM
matrix-matrix solve, inv(A) B
### 矩阵类型
GE
general   
GB
general band
SY
symmetric
SB
symmetric band
SP
symmetric packed
HE
hermitian
HB
hermitian band
HP
hermitian packed
TR
triangular
TB
triangular band
TP
triangular packed
Each operation is defined for four precisions,
### 数据类型
S
single real
D
double real
C
single complex
Z
double complex

Thus, for example, the name SGEMM stands for "single-precision general matrix-matrix multiply" and ZGEMM stands for "double-precision complex matrix-matrix multiply".

因此，例如，命名为SGEMM的函数意思为“单精度普通矩阵乘法”，ZGEMM为“双精度复数矩阵乘法”。

# CBLAS函数
Caffe中使用的blas函数是`cblas_*`，这里介绍常用的几个函数的使用方法：
## cblas_sgemm
`cblas_sgemm`为单精度实数普通矩阵乘法。
函数定义为：
```cpp
void cblas_sgemm(const enum CBLAS_ORDER Order, 
const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
const int M, const int N, const int K, 
const float alpha, 
const float *A, const int lda,
const float *B, const int ldb,
const float beta, 
float *C, const int ldc)
```
得到的结果是:
```cpp
 C = alpha*op(A)*op(B) + beta*C
```
其中`op(A)`是`M*K`维的矩阵，`op(B)`是`K*N`维的矩阵，`C`是`M*N`维的矩阵。
其他参数说明：
* `const enum CBLAS_ORDER Order`: 指数据的存储形式，在CBLAS的函数中无论一维还是二维数据都是用一维数组存储，这就要涉及是行主序还是列主序，在C语言中数组是用行主序，fortran中是列主序。用CblasRowMajor或CblasColMajor表示。
* `const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB`: 对矩阵A和B是否做转置操作，即`op(A),op(B)`的操作类型。
* `const int lda, const int ldb, const int ldc`: 对应A,B,C矩阵的列数。注意在BLAS的文档里，这三个参数分别为ABC的行数。