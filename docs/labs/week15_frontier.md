# Week 15 前沿专题

目标：把工业级推理系统论文机制映射到课程工程，完成一个小型改造或复现实验，并比较指标变化。

说明：cache-aware serving 等机制在本周以 teaching approximation 形式出现。示例 demo 用 shared-prefix score 说明请求重排的潜在价值，不代表完整在线 scheduler；报告需要说明和生产实现的差距。

## 代码入口

- `course_vllm/engine/policies.py`
- `course_vllm/benchmarks/cache_aware_demo.py`
- `docs/reports/week15_paper_to_system_template.md`

## 可选机制

- cache-aware serving
- prefill-decode disaggregation
- TokenDance-style scheduling

## 实验任务

1. 选择一个机制，生成 paper-to-system 映射。
2. 说明涉及的模块、数据结构、指标。
3. 运行最小复现实验，例如 cache-aware order 的 shared-prefix score。
4. 说明如何把机制从 demo 推进到完整服务主路径。

## 建议命令

```bash
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "cache-aware serving" \
  --prompts "1,2,3,4|1,2,3,9|8,7|1,2,5"
python -m course_vllm.benchmarks.grader week15
```

## 交付物

- paper-to-system 映射表。
- 改造前后指标。
- 对工程风险和下一步实现的说明。
