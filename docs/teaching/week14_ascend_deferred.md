# Week 14: AscendC

## 0. 本节学习目标

本节学习 AscendC：

- 了解华为 Ascend 的基本架构。
- 了解 AscendC 的编程方式。
- 对照 CUDA 理解两类加速卡编程模型的相似点和差异。
- 运行官方 Add 样例，理解 AscendC 算子开发的基本链路。
- 对照一个简单算子，比较 AscendC 版本与 CUDA 版本的差异。

这一节的重点是建立“同一个算子在不同 AI 加速器上如何表达”的对比视角。

## 1. 为什么要了解 AscendC

前面几周主要使用 CUDA。CUDA 是 NVIDIA GPU 的主流编程体系，很多 LLM serving 系统、kernel 库和 profiling 工具都围绕 CUDA 生态展开。但智能计算系统不只有 NVIDIA GPU，华为 Ascend NPU 也是常见 AI 加速器之一。

学习 AscendC 的目的不是在本课程里重写一套完整后端，而是建立一个对比视角：

```text
同样是 AI 加速器编程，
为什么 CUDA 代码不能直接搬到 Ascend 上？
为什么同一个 Add、RMSNorm、softmax 算子，
换硬件后要重新考虑数据搬运、并行粒度和内存层次？
```

这对理解“算子优化不是只写公式”很重要。一个算子的数学形式可能很简单，但高性能实现取决于硬件结构、存储层次、编程模型和工具链。

## 2. Ascend、CANN 与 AscendC 的基本概念

Ascend 是华为的 AI 处理器/加速卡体系。和 CUDA 对应 NVIDIA GPU 生态类似，Ascend 也有自己的软件栈和算子开发方式。

常见概念包括：

- Ascend AI Processor / NPU: 执行 AI 计算的硬件。
- AI Core: 执行算子计算的核心单元。
- CANN: Compute Architecture for Neural Networks，Ascend 的异构计算软件栈，提供编译、运行、算子库等能力。
- AscendC: 面向 Ascend 算子开发的 C/C++ 风格编程方式，用来编写自定义算子。

可以把关系粗略理解为：

```text
CUDA C++       -> NVIDIA GPU 自定义 kernel
AscendC        -> Ascend NPU 自定义算子
cuDNN/cuBLAS   -> NVIDIA 生态中的高性能库
CANN 算子库    -> Ascend 生态中的算子和运行支持
```

这个类比只是帮助入门，不能认为二者 API 或执行模型一一对应。

## 3. 和 CUDA 对照看编程模型

CUDA 中，一个最小 kernel 通常包含：

```text
host 侧:
  准备输入输出 tensor
  设置 grid/block
  launch kernel

device 侧:
  每个 thread 计算自己的 idx
  从 global memory 读数据
  执行计算
  写回 global memory
```

AscendC 也需要区分主机侧调度和设备侧计算，但术语、编译工具、内存层次和并行抽象与 CUDA 不同。学习时不要机械寻找一一对应的 `blockIdx`、`threadIdx`，而要问更本质的问题：

- 数据从哪里来，放到哪里？
- 一个计算任务被切成多少份？
- 每份由哪个计算单元执行？
- 片上存储如何使用？
- 什么时候需要同步？
- 如何把结果写回全局内存？

这和前面 CUDA 周次是同一类问题，只是硬件和 API 不同。

## 4. Add 样例

Add 也就是向量加法：

```text
out[i] = a[i] + b[i]
```

这个算子非常简单，但它适合作为任何新后端的第一步，因为它能验证完整链路：

```text
编译工具链是否可用
输入输出内存是否正确
host 侧是否能调用 device 算子
device 侧是否能并行读写
结果是否能和 reference 对齐
```

这和 Week03 的 CUDA vector add 是同一个教学目的。区别在于 Week03 用 CUDA 和 PyTorch extension，Week14 如果有 Ascend 环境，则用 AscendC 官方样例和 Ascend 工具链。

## 5. 如果比较 CUDA 与 AscendC，应该比较什么

如果有环境完成一个 AscendC 小算子，不应该只比较“代码长得像不像”。更有意义的是比较这些维度：

| 维度 | CUDA 中关注什么 | AscendC 中关注什么 |
| --- | --- | --- |
| 编译工具链 | nvcc、PyTorch extension | CANN/AscendC 编译链 |
| 并行组织 | grid、block、thread、warp | AscendC/AI Core 对应的任务组织方式 |
| 内存层次 | global/shared/register | 全局内存、片上缓存/本地存储等 |
| 数据搬运 | global load/store、coalescing | Ascend 上的数据搬运和对齐要求 |
| 正确性 | 和 PyTorch/CPU reference 对比 | 和同一个 reference 对比 |
| 性能 | ncu/nsys 或 CUDA event | Ascend 对应 profiling 工具 |

这样比较才贴近“智能计算系统编程与优化”的目标。

## 6. 如何学习 AscendC

学习 AscendC 时，可以沿着和 CUDA 入门相同的问题走：

- host 侧如何准备输入输出？
- device 侧如何描述并行计算？
- 数据如何从全局内存进入片上存储？
- 算子如何编译、注册和调用？
- 结果如何和同一个 reference 对齐？
- profiling 工具如何判断时间花在哪里？

如果有 Ascend 硬件和 CANN 工具链，可以运行官方 Add 样例，观察编译、运行和结果校验流程。没有 Ascend 环境时，也应完成 CUDA 与 AscendC 编程模型的对照理解：host/device 分工、并行粒度、内存层次、数据搬运和 profiling 工具分别如何变化。

## 7. 本节应该形成的对比视角

重点是理解：如果同样的 vector add 或 RMSNorm 放到 AscendC 上，数学公式不变，但编译工具链、设备侧并行方式、内存层次和 profiling 工具都会变化。
