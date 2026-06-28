# Week 16 总结与展示

目标：展示完整简化版 LLM 推理引擎，串联算子、缓存、调度、服务、性能分析和故障诊断。

## 展示内容

- 服务启动与 streaming 输出。
- CUDA kernel correctness 和 profiler 证据。
- KV cache/paged KV/prefix cache 示例。
- continuous batching 和 HTTP benchmark。
- week12 系统优化报告。
- week13 容量规划报告。
- week15 paper-to-system 映射。

## TODO(lab16)

- No new feature code required.
- Assemble: `docs/reports/week16_final_report_template.md`。
- Include evidence: tests/grader、profiling、benchmark、capacity planning、paper-to-system map。

## 建议命令

```bash
python -m pytest -q -rs
python -m course_vllm.benchmarks.grader week15
python -m course_vllm.benchmarks.capacity_planner --gpu-memory-gb 24 --weight-memory-gb 2 --num-layers 28 --num-kv-heads 8 --head-dim 128 --report
```

## 交付物

- 最终综合实验报告。
- 性能证据截图或摘要。
- 优化前后对比。
- 一次故障诊断复盘：问题、定位证据、修复、验证。
