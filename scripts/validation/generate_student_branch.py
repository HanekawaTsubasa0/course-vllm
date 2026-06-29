from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement + text[end_index + len(end) :]


def replace_function(text: str, start: str, end: str, body: str) -> str:
    return replace_between(text, start, end, body)


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text)


def read(path: str) -> str:
    return (ROOT / path).read_text()


def stub_python() -> None:
    files: dict[str, list[tuple[str, str, str]]] = {
        "course_vllm/engine/kv_cache.py": [
            (
                "    def append(self, seq_id: int, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:\n",
                "    def get(self, seq_id: int, layer_id: int) -> LayerKV:\n",
                """    def append(self, seq_id: int, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        \"\"\"TODO(lab08): append K/V tensors on the sequence dimension.\"\"\"
        raise NotImplementedError("TODO(lab08): implement ContinuousKVCache.append")

""",
            ),
            (
                "    def release(self, seq_id: int) -> None:\n",
                "    def num_layers_for(self, seq_id: int) -> int:\n",
                """    def release(self, seq_id: int) -> None:
        \"\"\"TODO(lab08): release all cached layers for one sequence.\"\"\"
        raise NotImplementedError("TODO(lab08): implement ContinuousKVCache.release")

""",
            ),
        ],
        "course_vllm/engine/block_manager.py": [
            (
                "    def ensure_capacity(self, seq_id: int, new_length: int) -> BlockTable:\n",
                "    def append_tokens(self, seq_id: int, num_new_tokens: int) -> BlockTable:\n",
                """    def ensure_capacity(self, seq_id: int, new_length: int) -> BlockTable:
        \"\"\"TODO(lab10): allocate enough physical blocks for new_length tokens.\"\"\"
        raise NotImplementedError("TODO(lab10): implement BlockManager.ensure_capacity")

""",
            ),
            (
                "    def slot_mapping(self, seq_id: int, positions: list[int]) -> list[int]:\n",
                "    def release(self, seq_id: int) -> None:\n",
                """    def slot_mapping(self, seq_id: int, positions: list[int]) -> list[int]:
        \"\"\"TODO(lab10): map logical token positions to physical KV slots.\"\"\"
        raise NotImplementedError("TODO(lab10): implement BlockManager.slot_mapping")

""",
            ),
            (
                "    def _allocate_with_prefix_cache(self, table: BlockTable, token_ids: list[int]) -> None:\n",
                "    def _allocate_fresh_block(self) -> int:\n",
                """    def _allocate_with_prefix_cache(self, table: BlockTable, token_ids: list[int]) -> None:
        \"\"\"TODO(lab10): reuse full prefix blocks via hash and reference counts.\"\"\"
        raise NotImplementedError("TODO(lab10): implement prefix-cache block allocation")

""",
            ),
        ],
        "course_vllm/engine/paged_kv_cache.py": [
            (
                "    def reserve(self, seq_id: int, num_new_tokens: int) -> list[int]:\n",
                "    def write(\n",
                """    def reserve(self, seq_id: int, num_new_tokens: int) -> list[int]:
        \"\"\"TODO(lab10): reserve logical positions and grow the block table.\"\"\"
        raise NotImplementedError("TODO(lab10): implement PagedKVCache.reserve")

""",
            ),
            (
                "    def write(\n",
                "    def get_dense(self, seq_id: int, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:\n",
                """    def write(
        self,
        seq_id: int,
        layer_id: int,
        positions: list[int],
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        skip_shared: bool = False,
    ) -> None:
        \"\"\"TODO(lab10): write per-token K/V into physical slots.\"\"\"
        raise NotImplementedError("TODO(lab10): implement PagedKVCache.write")

""",
            ),
        ],
        "course_vllm/engine/request.py": [
            (
                "    def append_token(self, token_id: int) -> None:\n",
                "    def scheduled_prompt_tokens(self) -> list[int]:\n",
                """    def append_token(self, token_id: int) -> None:
        \"\"\"TODO(lab09): append one generated token to the sequence state.\"\"\"
        raise NotImplementedError("TODO(lab09): implement Sequence.append_token")

""",
            ),
            (
                "    def finish(self) -> None:\n",
                "",
                """    def finish(self) -> None:
        \"\"\"TODO(lab09): mark both sequence and request as finished.\"\"\"
        raise NotImplementedError("TODO(lab09): implement Sequence.finish")
""",
            ),
        ],
        "course_vllm/engine/scheduler.py": [
            (
                "    def add(self, seq: Sequence) -> None:\n",
                "    def schedule(self) -> ScheduledBatch | None:\n",
                """    def add(self, seq: Sequence) -> None:
        \"\"\"TODO(lab11): put a new sequence into the waiting queue.\"\"\"
        raise NotImplementedError("TODO(lab11): implement Scheduler.add")

""",
            ),
            (
                "    def _schedule_prefill(self) -> ScheduledBatch | None:\n",
                "    def _schedule_decode(self) -> ScheduledBatch | None:\n",
                """    def _schedule_prefill(self) -> ScheduledBatch | None:
        \"\"\"TODO(lab11): build a prefill batch under sequence/token budgets.\"\"\"
        raise NotImplementedError("TODO(lab11): implement Scheduler._schedule_prefill")

""",
            ),
            (
                "    def _schedule_decode(self) -> ScheduledBatch | None:\n",
                "    def finish(self, seq: Sequence) -> None:\n",
                """    def _schedule_decode(self) -> ScheduledBatch | None:
        \"\"\"TODO(lab11): select running sequences for one-token decode.\"\"\"
        raise NotImplementedError("TODO(lab11): implement Scheduler._schedule_decode")

""",
            ),
        ],
        "course_vllm/engine/sampler.py": [
            (
                "    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:\n",
                "",
                """    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        \"\"\"TODO(lab06): implement stable softmax and optionally dispatch CUDA softmax.\"\"\"
        raise NotImplementedError("TODO(lab06): implement Sampler._softmax")
""",
            ),
        ],
        "course_vllm/model/ops.py": [
            (
                "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
                "\n\ndef dense_attention_prefill_reference",
                """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"TODO(lab05): dispatch to the teaching matmul kernel and keep torch fallback.\"\"\"
        raise NotImplementedError("TODO(lab05): implement CourseLinear.forward")


""",
            ),
            (
                "def dense_attention_prefill(\n",
                "\n\ndef dense_attention_decode_reference",
                """def dense_attention_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    kernel_impl: str = "torch",
    block_size: int | None = None,
) -> torch.Tensor:
    \"\"\"TODO(lab07): implement causal prefill attention dispatch.\"\"\"
    raise NotImplementedError("TODO(lab07): implement dense_attention_prefill")


""",
            ),
        ],
        "course_vllm/model/attention.py": [
            (
                "def paged_attention_decode(\n",
                "\n\ndef paged_attention_decode_reference",
                """def paged_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: Sequence[Sequence[int]] | torch.Tensor,
    context_lens: Sequence[int] | torch.Tensor,
    block_size: int,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    \"\"\"TODO(lab07): dispatch paged decode attention to CUDA or reference.\"\"\"
    raise NotImplementedError("TODO(lab07): implement paged_attention_decode")


""",
            ),
        ],
        "course_vllm/model/qwen3_torch.py": [
            (
                "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
                "\n\ndef rotate_half",
                """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"TODO(lab04): implement RMSNorm and optional CUDA dispatch.\"\"\"
        raise NotImplementedError("TODO(lab04): implement Qwen3RMSNorm.forward")


""",
            ),
            (
                "def apply_rotary_pos_emb(\n",
                "\n\ndef _cuda_rope_nd",
                """def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    kernel_impl: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"TODO(lab04): apply RoPE to Q and K, with optional CUDA dispatch.\"\"\"
    raise NotImplementedError("TODO(lab04): implement apply_rotary_pos_emb")


""",
            ),
        ],
        "course_vllm/model/qwen3_continuous_backend.py": [
            (
                "    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:\n",
                "",
                """    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        \"\"\"TODO(lab12): add pinned-memory and transfer-stream optimized CPU->GPU copy.\"\"\"
        if self.device.type != "cuda":
            return tensor.to(self.device)
        return tensor.to(self.device)
""",
            ),
        ],
        "course_vllm/server/batching.py": [
            (
                "    def _admit(self, prompt: str) -> None:\n",
                "    async def _run_model",
                """    def _admit(self, prompt: str) -> None:
        \"\"\"TODO(lab12): enforce prompt length and queue depth admission limits.\"\"\"
        raise NotImplementedError("TODO(lab12): implement BatchingEngine._admit")

""",
            ),
        ],
    }
    for path, replacements in files.items():
        text = read(path)
        for start, end, body in replacements:
            if end:
                text = replace_function(text, start, end, body)
            else:
                start_index = text.index(start)
                text = text[:start_index] + body
        write(path, text)


def stub_cuda() -> None:
    write(
        "kernels/vector_add.cu",
        """#include <torch/extension.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO(lab03): write out[idx] = a[idx] + b[idx] when idx is in range.
  if (idx < n) out[idx] = 0.0f;
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "inputs must be float32");
  TORCH_CHECK(a.numel() == b.numel(), "input sizes must match");
  auto out = torch::empty_like(a);
  int n = a.numel();
  vector_add_kernel<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "vector_add");
}
""",
    )
    text = read("kernels/course_ops.cu")
    replacements = [
        (
            "template <typename scalar_t>\n__global__ void softmax_kernel",
            "template <typename scalar_t>\n__global__ void rms_norm_kernel",
            """template <typename scalar_t>
__global__ void softmax_kernel(const scalar_t* x, scalar_t* out, int rows, int cols) {
  // TODO(lab06): implement row-wise numerically stable softmax.
  int row = blockIdx.x;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[row * cols + col] = from_float<scalar_t>(0.0f);
  }
}

""",
        ),
        (
            "template <typename scalar_t>\n__global__ void rms_norm_kernel",
            "template <typename scalar_t>\n__global__ void rope_kernel",
            """template <typename scalar_t>
__global__ void rms_norm_kernel(const scalar_t* x, const scalar_t* weight, scalar_t* out, int rows, int cols, float eps) {
  // TODO(lab04): compute RMSNorm over each row.
  int row = blockIdx.x;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[row * cols + col] = x[row * cols + col];
  }
}

""",
        ),
        (
            "template <typename scalar_t>\n__global__ void rope_kernel",
            "template <typename scalar_t>\n__global__ void matmul_kernel",
            """template <typename scalar_t>
__global__ void rope_kernel(const scalar_t* x, const scalar_t* cos, const scalar_t* sin, scalar_t* out, int rows, int cols) {
  // TODO(lab04): implement rotate-half RoPE.
  int row = blockIdx.x;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[row * cols + col] = x[row * cols + col];
  }
}

""",
        ),
        (
            "template <typename scalar_t>\n__global__ void matmul_kernel",
            "template <typename scalar_t, int Tile>\n__global__ void matmul_tiled_kernel",
            """template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int m, int n, int k) {
  // TODO(lab05): implement naive C = A @ B.
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) out[row * n + col] = from_float<scalar_t>(0.0f);
}

""",
        ),
        (
            "template <typename scalar_t, int Tile>\n__global__ void matmul_tiled_kernel",
            "template <typename scalar_t>\n__global__ void paged_attention_decode_kernel",
            """template <typename scalar_t, int Tile>
__global__ void matmul_tiled_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int m, int n, int k) {
  // TODO(lab05): implement tiled matmul using shared memory.
  int row = blockIdx.y * Tile + threadIdx.y;
  int col = blockIdx.x * Tile + threadIdx.x;
  if (row < m && col < n) out[row * n + col] = from_float<scalar_t>(0.0f);
}

""",
        ),
        (
            "template <typename scalar_t>\n__global__ void paged_attention_decode_kernel",
            "template <typename scalar_t>\n__global__ void dense_attention_prefill_kernel",
            """template <typename scalar_t>
__global__ void paged_attention_decode_kernel(
    const scalar_t* query,
    const scalar_t* key_cache,
    const scalar_t* value_cache,
    const int64_t* block_tables,
    const int64_t* context_lens,
    scalar_t* out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks,
    int block_size,
    float scale) {
  // TODO(lab07): implement paged decode attention over block tables.
  int batch = blockIdx.x;
  int head = blockIdx.y;
  int dim = threadIdx.x;
  if (batch < batch_size && head < num_heads && dim < head_dim) {
    out[(batch * num_heads + head) * head_dim + dim] = from_float<scalar_t>(0.0f);
  }
}

""",
        ),
        (
            "template <typename scalar_t>\n__global__ void dense_attention_prefill_kernel",
            "template <typename scalar_t>\n__global__ void dense_attention_decode_kernel",
            """template <typename scalar_t>
__global__ void dense_attention_prefill_kernel(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    scalar_t* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {
  // TODO(lab07): implement causal prefill attention.
  int batch = blockIdx.x;
  int head = blockIdx.y;
  int query_pos = blockIdx.z;
  int dim = threadIdx.x;
  if (batch < batch_size && head < num_heads && query_pos < seq_len && dim < head_dim) {
    out[((batch * num_heads + head) * seq_len + query_pos) * head_dim + dim] = from_float<scalar_t>(0.0f);
  }
}

""",
        ),
    ]
    for start, end, body in replacements:
        text = replace_between(text, start, end, body)
    write("kernels/course_ops.cu", text)


def update_docs() -> None:
    text = read("README.md")
    insert = """\n## Student Starter Notes\n\nThis branch is the student starter version. Core lab code intentionally contains `TODO(labXX)` stubs. Use `--backend reference` for a runnable oracle while implementing course modules, and run the week-specific grader as each lab is completed.\n\n```bash\npython -m course_vllm.server.api --backend reference --stage week01 --port 18080\npython -m course_vllm.benchmarks.grader week03\n```\n\nThe full solution remains on `main`.\n"""
    if "## Student Starter Notes" not in text:
        text = text.replace("## 环境配置\n", insert + "\n## 环境配置\n")
    write("README.md", text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the student starter branch from main.")
    parser.add_argument("--branch", default="student")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise SystemExit("working tree must be clean before generating the student branch")
    if args.force:
        run(["git", "branch", "-D", args.branch])
    run(["git", "switch", "-c", args.branch])
    stub_python()
    stub_cuda()
    update_docs()
    run(["git", "add", "."])
    run(["git", "commit", "-m", "Create student starter branch"])


if __name__ == "__main__":
    main()
