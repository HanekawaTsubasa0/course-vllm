# Week 03: CUDA Extension 入门

## 0. 本节学习目标

Week03 只讲 CUDA 编程模型和 PyTorch CUDA extension 的最小闭环，不讲复杂模型算子。目标是从 Python 世界进入 GPU kernel 世界：知道一个 tensor 怎么穿过 Python wrapper、C++ binding、CUDA launcher，最后由成百上千个 GPU thread 并行处理。

本周重点不是性能优化，而是正确理解 CUDA execution model。

## 1. 为什么要写自定义 CUDA 算子

PyTorch 已经提供了很多高性能算子，例如 matmul、softmax、layer norm。那为什么还要学习写 CUDA？

原因有三个。

第一，serving 系统里经常需要特殊数据布局和特殊 kernel。PagedAttention 就是典型例子：K/V 不是连续存放，而是通过 block table 查找。通用 PyTorch attention 不一定能表达这种访问模式。

第二，理解 CUDA 有助于理解性能瓶颈。即使最终使用 cuBLAS、FlashAttention 或 vendor kernel，也需要知道 memory coalescing、shared memory、occupancy、warp divergence 这些概念，否则看不懂 profiler。

第三，学习上需要把“模型公式”落到“硬件执行”。只有写过最小 kernel，才能理解后续 RMSNorm、softmax、matmul、attention 为什么要按 thread/block 组织。

## 2. CUDA execution model

CUDA kernel 是运行在 GPU 上的函数。一次 kernel launch 会启动一个 grid。grid 由多个 block 组成，block 由多个 thread 组成。

一维 kernel 最常见的索引方式是：

```text
idx = blockIdx.x * blockDim.x + threadIdx.x
```

含义是：

- `threadIdx.x`: 当前线程在 block 内的编号。
- `blockIdx.x`: 当前 block 在 grid 内的编号。
- `blockDim.x`: 每个 block 的线程数。
- `idx`: 当前线程负责的全局元素编号。

假设 `n = 1024`，`blockDim.x = 256`，那么需要 4 个 block，总共 1024 个线程。每个线程处理一个元素。

如果 `n = 1000`，仍然需要 4 个 block，总线程数 1024。多出来的 24 个线程必须什么都不做，所以需要：

```text
if idx < n:
    ...
```

这叫 boundary check。没有边界检查就可能写越界。

## 3. CUDA 内存层次的第一印象

刚开始写 CUDA 时，先认识这些内存层次：

- global memory: GPU 显存，容量大，访问慢。
- shared memory: 一个 block 内线程共享，容量小，访问快。
- registers: 每个线程自己的寄存器，最快但数量有限。
- constant memory / texture memory: 特殊只读访问场景。

vector add 只用 global memory。每个线程读 `a[idx]` 和 `b[idx]`，写 `out[idx]`。它几乎没有数据复用，因此主要是 memory bandwidth 练习。

## 4. PyTorch CUDA extension 的调用链

课程实验使用 PyTorch extension，而不是单独写 `.cu` 程序。原因是后续模型代码都使用 PyTorch tensor。自定义 kernel 必须能接受 PyTorch tensor，返回 PyTorch tensor，并参与 Python 侧测试。

典型调用链：

```text
Python test
-> load_cuda_extension(...)
-> torch.utils.cpp_extension.load
-> nvcc 编译 .cu
-> pybind11 暴露 C++/CUDA 函数
-> Python 调用 module.vector_add(a, b)
-> CUDA kernel launch
```

`pybind11` 的作用是把 C++ 函数暴露给 Python。PyTorch extension 还负责把 `torch.Tensor` 对象传进 C++，让 C++ 侧可以检查 device、dtype、shape，并拿到底层 data pointer。

## 5. vector add 为什么适合做第一课

vector add 的数学非常简单：

```text
out[i] = a[i] + b[i]
```

它没有 reduction，没有 shared memory，没有复杂数据布局。这样可以把注意力集中在：

- kernel 函数怎么写。
- thread id 怎么计算。
- grid/block 怎么设置。
- 边界检查怎么做。
- Python 如何调用 CUDA。
- 测试如何和 PyTorch reference 对齐。

这相当于后续所有 CUDA lab 的最小原型。

## 6. 从 Python 调到 CUDA

PyTorch CUDA extension 的核心价值是把 Python tensor 交给自定义 CUDA kernel。典型链路是：

```text
Python function
-> C++ binding
-> CUDA launcher
-> CUDA kernel
-> output tensor
```

Python 层负责组织输入和测试，C++ binding 负责把 PyTorch tensor 传给底层函数，CUDA launcher 负责设置 grid/block 并启动 kernel，kernel 负责真正的并行计算。

vector add 的输入输出 shape 很简单：

```text
a: [n]
b: [n]
out: [n]
```

测试时通常用 PyTorch 的 `a + b` 作为 reference。自定义 CUDA 输出只要和 reference 在误差范围内一致，就说明最小闭环跑通了。

## 7. 写完 vector add 后应该理解什么

需要实现：

```text
idx = blockIdx.x * blockDim.x + threadIdx.x
if idx < n:
    out[idx] = a[idx] + b[idx]
```

这一节先把注意力放在 kernel body：线程如何定位元素、如何做边界检查、如何写回输出。C++ binding 和 Python harness 先作为调用框架来理解。

## 8. 实验中的少量对照

实验会让一个 CUDA kernel 完成向量加法，并用 PyTorch reference 做正确性比较。这里的重点不是向量加法本身，而是建立后续所有自定义算子的基本开发流程：写 kernel、编译 extension、从 Python 调用、对齐 reference、测量耗时。
