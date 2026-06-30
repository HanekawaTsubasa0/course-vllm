# Week 12: 系统优化与 Admission Control

## 0. 本节学习目标

Week12 聚焦模型以外的系统优化：CPU 到 GPU 的数据传输、pinned memory、CUDA stream、transfer stream、admission control。它不再讲新的 transformer 算子，而是讲 serving 系统如何保护自身和减少非模型开销。

## 1. Host-to-device copy

模型权重通常已经在 GPU 上，但请求 token、attention mask、position ids 等输入数据可能从 CPU 构造后拷到 GPU。这个过程叫 host-to-device copy, H2D copy。

如果每次 copy 都同步阻塞，CPU 和 GPU 之间会出现等待。理想情况下，数据传输可以和部分计算重叠。

在 LLM serving 中，H2D copy 往往不是最大 FLOPs 开销，但它容易制造 timeline 上的空洞。尤其是在小 batch、短 prompt、decode step 很小的场景里，GPU kernel 本身很短，CPU 准备输入和数据传输的相对占比会变大。此时单纯优化模型 kernel 可能看不到明显收益，因为系统瓶颈已经移动到数据准备和调度。

学习时要注意：系统优化不是“所有开关都打开就一定更快”。它依赖 workload。长 prompt、大 batch 时，模型计算可能淹没 copy 成本；短请求、高并发时，copy、调度和同步可能显得更重要。

## 2. Pinned memory

普通 CPU 内存可能被操作系统分页移动。GPU DMA 更喜欢页锁定内存，也就是 pinned memory。pinned memory 不能被操作系统随意换出，因此可以支持更高效的异步传输。

使用 pinned memory 的代价是它会占用宝贵的页锁定资源，过度使用会影响系统。它不是所有 workload 都一定更快，但在高频 H2D copy 场景中很重要。

PyTorch 中常见用法是先把 CPU tensor 放到 pinned memory，再用 non-blocking copy 传到 GPU：

```text
cpu_tensor.pin_memory()
gpu_tensor = cpu_tensor.to(device="cuda", non_blocking=True)
```

但 `non_blocking=True` 不等于一定异步。它需要 pinned memory、合适的 stream、没有立即消费导致的同步等条件共同成立。否则代码看起来是异步，timeline 上仍然可能串行。

## 3. CUDA stream 和 transfer stream

CUDA stream 是 GPU 操作的有序队列。同一个 stream 内操作按顺序执行，不同 stream 之间可能并发或重叠。

如果把 H2D copy 放到专门的 transfer stream，同时 compute 在另一个 stream 上运行，理论上可以实现传输和计算重叠。但是否真正重叠取决于硬件、依赖关系、tensor 生命周期和同步点。

这就是为什么系统优化必须配合 Nsight Systems 看 timeline。

一个常见错误是：创建了 transfer stream，但马上在 default stream 上使用刚拷贝的 tensor，并隐式等待 copy 完成。这样程序语义正确，但没有得到预期重叠。真正的重叠需要合理安排依赖，让 GPU 在 copy 另一个 batch 输入时，仍能计算当前 batch。

判断异步是否真的生效，不能只看代码里有没有 `non_blocking=True` 或 stream 对象。要看 timeline：

```text
copy batch N+1
overlaps with
compute batch N
```

如果 copy 和 compute 在时间线上仍然前后串行，就说明依赖、同步点或内存条件没有满足。

## 4. Admission control

admission control 是请求进入系统前的保护机制。LLM 服务里常见限制包括：

- 最大 prompt 长度。
- 最大队列长度。
- 最大 batch size。
- 最大 batched tokens。
- 最大并发请求。

没有 admission control 时，过长 prompt 或过多请求可能导致队列无限增长、TTFT 爆炸、显存耗尽。拒绝请求有时比让所有请求超时更好。

admission control 的本质是把系统容量边界显式化。在线服务不应该假设所有请求都能被无限接收。尤其是 LLM 请求成本差异很大：一个 20 token prompt 和一个 20k token prompt 对 prefill、KV cache 和排队的影响完全不同。

常见策略包括：

- prompt token 数限制。
- 请求队列长度限制。
- 单用户并发限制。
- 预计 KV cache 占用限制。
- 按优先级接收或拒绝。

入门实现通常会先做 prompt 长度和队列深度限制。生产系统会进一步用 tokenizer 后的 token 数、预计输出长度和当前 KV cache 剩余容量做判断。

## 5. 投机解码与量化的方向性理解

投机解码 speculative decoding 是一种降低 decode 成本的思路。它通常使用一个较小、较快的 draft model 先提出若干候选 token，再由目标大模型一次性验证这些 token。若候选被接受，就能减少大模型逐 token 调用的次数；若候选被拒绝，则需要回退到大模型结果。

投机解码的关键不是“让小模型替代大模型”，而是“用小模型猜测，大模型保证分布正确或近似正确”。它适合大模型 decode 成本高、draft model 足够快且接受率较高的场景。工程上要处理 draft/verify 两套模型、KV cache 同步、接受率统计和回退逻辑。

量化 quantization 是减少权重、activation 或 KV cache 数值精度的方法，例如从 FP16/BF16 降到 INT8、INT4 或 FP8。它的收益通常包括更低显存占用、更高带宽效率和可能更高吞吐；风险是精度下降、kernel 更复杂、不同硬件支持差异明显。

这一节把投机解码和量化作为方向性知识理解。它们和 pinned memory、stream、admission control 一样，都是从系统层面降低成本或保护服务的手段。

## 6. 做性能对比时要注意什么

系统优化必须用对照实验判断收益。只打开 pinned memory 或 transfer stream，并不自动说明性能变好。要比较优化前后：

- TTFT 是否降低。
- TPOT 是否降低。
- output tokens/s 是否提高。
- GPU timeline 是否减少空洞。
- H2D copy 是否和 compute 有重叠。
- admission control 是否减少超长排队或 OOM。

还要注意 workload。长 prompt、大 batch 时，模型计算可能是主导；短请求、高并发时，CPU 调度、H2D copy 和同步开销更容易显现。

系统优化的正确姿势是一次只改变一个主要因素。否则看到性能变化时，很难判断是 pinned memory、生效的 stream、batch size 变化，还是请求分布变化导致的。

## 7. 实验中的少量对照

实验会把 pinned memory、transfer stream、队列上限和 prompt 长度限制作为可观察开关。阅读代码时重点看这些开关如何影响数据传输、请求接收和健康状态统计。
