# Week 05: Matmul 与 Linear

## 0. 本节学习目标

Week05 聚焦矩阵乘和线性层。前面 Week04 讲的是逐行归一化和位置旋转，这些算子很重要但不是 LLM 计算量的主体。真正占据 transformer 大量 FLOPs 的，是 attention projection、MLP projection 中的 linear，也就是矩阵乘。

本周要讲清楚 GEMM、weight layout、naive matmul、tiled matmul 和 shared memory 的基本思想。

## 1. Linear 为什么核心

Transformer 层里到处都是线性变换：

- attention 的 Q projection。
- attention 的 K projection。
- attention 的 V projection。
- attention output projection。
- MLP 的 gate/up/down projection。

数学上，linear layer 是：

```text
y = x W^T + b
```

如果忽略 bias，就是矩阵乘。假设：

```text
x: [tokens, in_features]
weight: [out_features, in_features]
y: [tokens, out_features]
```

那么计算等价于：

```text
y = x @ weight.T
```

LLM 中 hidden size 和 intermediate size 都很大，因此 GEMM 是主要计算来源之一。

## 2. Naive matmul

矩阵乘：

```text
C[M, N] = A[M, K] @ B[K, N]
```

每个输出元素：

```text
C[i, j] = sum_{k=0}^{K-1} A[i, k] * B[k, j]
```

最直观的 CUDA 实现是每个线程计算一个 `C[i, j]`。线程根据二维 block/grid 得到 `i` 和 `j`，然后在 K 维循环。

这种实现容易理解，但性能不高。原因是 A/B 元素会被反复从 global memory 读取。例如同一行 A 会被多个输出列使用，同一列 B 会被多个输出行使用。naive 实现没有利用这种数据复用。

## 3. Tiled matmul 和 shared memory

tiled matmul 的核心思想是把 A 和 B 分块。一个 thread block 负责 C 的一个 tile。它先把 A 的 tile 和 B 的 tile 从 global memory 读到 shared memory，然后 block 内线程重复使用 shared memory 中的数据。

流程大致是：

```text
for tile_k in K dimension:
    load A tile into shared memory
    load B tile into shared memory
    __syncthreads()
    compute partial sum
    __syncthreads()
write C
```

shared memory 比 global memory 快，但容量小，只在一个 block 内共享。tiled matmul 用 shared memory 换取 global memory 访问减少。

这个机制是 CUDA 优化的经典例子：不是改变数学公式，而是改变数据搬运方式。

## 4. 工程上还要注意什么

第一，weight layout 容易弄错。PyTorch `nn.Linear` 的 weight 通常是 `[out_features, in_features]`，而矩阵乘需要 B 是 `[in_features, out_features]`。所以实现时经常要转置或按转置后的逻辑访问。

第二，边界 tile 必须处理。M、N、K 不一定是 tile size 的整数倍。

第三，浮点误差不可避免。不同线程累加顺序不同，结果可能和 PyTorch 有微小差异，所以测试用 `allclose` 而不是逐 bit 相等。

第四，教学中的基础 matmul 不等于工业 GEMM。cuBLAS、CUTLASS 会使用更复杂的 tiling、tensor cores、warp-level MMA、pipeline 等技巧。学习时先掌握最基础的 shared memory tiling，再理解工业库为什么复杂。

## 5. 实现时最容易错的地方

naive matmul 最容易错的是索引。给定：

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

输出 `C[i, j]` 应该累加：

```text
sum_k A[i, k] * B[k, j]
```

tiled matmul 最容易错的是 tile 边界和同步。一个 tile 加载到 shared memory 后，block 内线程必须在计算前同步；计算完当前 tile 后，再进入下一个 tile。边界 tile 不满时，越界元素要当作 0 处理。

## 6. 实验中的少量对照

实验会先实现 naive matmul，再实现 shared-memory tiled matmul，并把它接到 linear 层。阅读代码时只需要确认三件事：shape 是否对齐、weight layout 是否正确、CUDA 输出是否和 PyTorch reference 接近。
