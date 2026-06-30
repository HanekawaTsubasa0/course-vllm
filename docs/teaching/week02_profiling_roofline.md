# Week 02: Profiling 与性能分析

## 0. 本节学习目标

Week02 不继续展开 LLM serving 的完整流程，而是专门讲“如何判断系统慢在哪里”。学习重点是从第一周的功能性视角切换到性能分析视角：同样是一次生成请求，它可能慢在 CPU、GPU、内存拷贝、kernel 本身、队列等待、batching 策略、同步点或 HTTP 客户端。没有 profiling，就只能凭感觉猜。

本周的核心能力是：看到一个性能指标变化后，能提出合理假设，并知道用哪种工具验证。

## 1. 性能分析为什么不能只看总耗时

LLM serving 的一次请求跨越很多层：

```text
client
-> HTTP server
-> request queue
-> tokenizer
-> model forward
-> CUDA kernels
-> KV cache read/write
-> sampler
-> response serialization
```

如果只看到 `latency=2s`，信息是不够的。可能是 prompt 很长导致 prefill 慢，也可能是 decode token 多，也可能是请求排队，也可能是 CUDA kernel launch 间隔大，也可能是 CPU 正在等待 GPU 同步。优化前必须先定位瓶颈层次。

性能分析通常分三层：

服务层指标回答“用户和系统看到了什么”：

- requests/s
- output tokens/s
- TTFT
- TPOT
- p50/p90/p99 latency
- queue depth
- average batch size

框架层 profiler 回答“PyTorch 和 CUDA op 花了多少时间”：

- 哪些 op 时间最长。
- 是否有大量 `aten::copy_`。
- 是否有 CPU 等 GPU。
- 是否有意外的同步。

kernel 层 profiler 回答“单个 CUDA kernel 为什么没有跑满”：

- memory throughput。
- achieved occupancy。
- warp stall 原因。
- shared memory 使用。
- global load/store efficiency。

这三层不能互相替代。服务层指标能说明用户体验，但不能解释 kernel 为什么慢；kernel profiler 能解释一个 kernel，但不能说明队列和 batching 是否合理。

## 2. Roofline 思维

Roofline model 是一种性能上界分析方法。它把程序性能限制粗略分成两类：compute-bound 和 memory-bound。

一个算子的 arithmetic intensity 定义为：

```text
arithmetic intensity = FLOPs / bytes moved
```

如果一个算子每读取很少数据就做很多计算，它更可能受计算峰值限制；如果它每做一点计算就要搬很多数据，它更可能受内存带宽限制。

例如矩阵乘 `C = A @ B`，当矩阵足够大且实现合理时，同一份 A/B 元素会被复用很多次，arithmetic intensity 可以很高，因此可能 compute-bound。RMSNorm、softmax 这类逐行归约算子往往需要读写整行数据，计算相对少，更容易 memory-bound。

学习 Roofline 时，不必一开始就精确算每个 kernel 的 FLOPs 和 bytes，更重要的是建立判断习惯：

- 如果一个算子大部分时间在读写内存，优化方向是减少访存、提高 coalescing、复用数据。
- 如果一个算子计算密集，优化方向是提高并行度、使用更合适的数据类型、调用高效 GEMM。
- 如果 GPU timeline 上有大量空洞，瓶颈可能不是 kernel 本身，而是调度、同步或 CPU 侧。

## 3. LLM serving 里的典型瓶颈

prefill 阶段通常有较大的矩阵乘和长序列 attention。它的 GPU 利用率可能比较高，但 prompt 很长时 TTFT 会变大。prefill 的优化方向包括 batch prefill、chunked prefill、attention kernel 优化和减少不必要的数据搬移。

decode 阶段每步只生成一个 token。单请求 decode 的计算粒度很小，GPU 容易吃不满。decode 的瓶颈常常来自 KV cache 读取、attention decode kernel、batch size 不足、调度开销。continuous batching 的主要价值就在 decode 阶段。

KV cache 会引入显存压力和访存压力。随着上下文长度增长，decode attention 要读取越来越多历史 K/V。即使计算量不大，读 KV 的内存带宽也可能成为瓶颈。

sampling 通常不是最大开销，但在 vocab 很大、batch 较大、top-k/top-p 复杂时也可能显著。为了理解完整请求路径，也要知道 logits 到 token 的过程也在请求路径里。

服务端队列会影响 TTFT 和 tail latency。吞吐优化通常会让请求等待合批窗口，这可能提升 tokens/s，但增加部分请求延迟。

## 4. 常用工具应该怎么看

PyTorch profiler 适合回答：

- 哪些 PyTorch op 或 CUDA kernel 占用时间最多。
- CPU 时间和 CUDA 时间是否匹配。
- 是否出现大量 copy、contiguous、reshape 之类的隐藏开销。
- 是否有意外同步。

Nsight Systems 适合回答：

- CPU thread 和 GPU stream 的时间线关系。
- GPU 是否有空洞。
- kernel launch 是否密集。
- memcpy 是否和 compute 重叠。
- 服务进程是否在等待 I/O 或 sleep。

Nsight Compute 适合回答：

- 某个 kernel 的 occupancy 是否低。
- global memory load/store 是否高效。
- shared memory 是否有 bank conflict。
- warp stall 的主要原因是什么。
- 实际带宽接近硬件上限多少。

服务端 benchmark 适合回答：

- 请求吞吐和 token 吞吐是多少。
- 合批前后吞吐变化多少。
- latency 分位数如何变化。
- tail latency 是否恶化。

## 5. 如何形成一个性能结论

性能分析不能只写“运行变慢了”或“GPU 利用率不高”。一个合格的性能结论至少包含四步。

第一，明确观测指标。比如 TTFT、TPOT、requests/s、tokens/s、p90 latency、GPU utilization、kernel duration、H2D copy 时间。

第二，提出瓶颈假设。比如“长 prompt 导致 prefill 时间变长”“decode batch 太小导致 GPU 吃不满”“某个 softmax kernel memory-bound”“CPU 同步导致 GPU 等待”。

第三，用工具验证。系统时间线工具适合看 CPU、GPU、拷贝和同步的先后关系；kernel 分析工具适合看单个 CUDA kernel 的访存、occupancy、warp stall 和算术利用率；服务压测工具适合看端到端吞吐和延迟分位数。

第四，说明证据边界。一次 profiler 采样不一定代表所有 workload；batch size、prompt 长度、输出长度、并发数、sampling 参数都会改变瓶颈。

学习记录里要区分：

- 观测到的指标是什么。
- 这些指标说明了什么现象。
- 现象可能由哪些瓶颈造成。
- 使用了哪个工具验证。
- 还有哪些证据不足。

## 6. 实验中的少量对照

实验中会运行服务压测、PyTorch profiler、Nsight Systems 和 Nsight Compute。使用这些工具时，重点不是记命令，而是把输出归类：端到端指标、框架 op 热点、系统时间线、单 kernel 细节分别回答哪类问题。
