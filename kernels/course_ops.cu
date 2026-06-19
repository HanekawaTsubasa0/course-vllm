#include <c10/cuda/CUDAException.h>
#include <ATen/Dispatch.h>
#include <torch/extension.h>

#include <cmath>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void softmax_kernel(const scalar_t* x, scalar_t* out, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* row_x = x + row * cols;
  float row_max = -INFINITY;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    row_max = fmaxf(row_max, static_cast<float>(row_x[col]));
  }
  __shared__ float shared[kThreads];
  shared[threadIdx.x] = row_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  row_max = shared[0];

  float denom = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    denom += expf(static_cast<float>(row_x[col]) - row_max);
  }
  shared[threadIdx.x] = denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  denom = shared[0];
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[row * cols + col] = static_cast<scalar_t>(expf(static_cast<float>(row_x[col]) - row_max) / denom);
  }
}

template <typename scalar_t>
__global__ void rms_norm_kernel(const scalar_t* x, const scalar_t* weight, scalar_t* out, int rows, int cols, float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* row_x = x + row * cols;
  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = static_cast<float>(row_x[col]);
    sum_sq += value * value;
  }
  __shared__ float shared[kThreads];
  shared[threadIdx.x] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  float scale = rsqrtf(shared[0] / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[row * cols + col] = static_cast<scalar_t>(static_cast<float>(row_x[col]) * scale * static_cast<float>(weight[col]));
  }
}

template <typename scalar_t>
__global__ void rope_kernel(const scalar_t* x, const scalar_t* cos, const scalar_t* sin, scalar_t* out, int rows, int cols) {
  int row = blockIdx.x;
  int half = cols / 2;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    int rotated_col = col < half ? col + half : col - half;
    float sign = col < half ? -1.0f : 1.0f;
    int idx = row * cols + col;
    out[idx] = static_cast<scalar_t>(
        static_cast<float>(x[idx]) * static_cast<float>(cos[idx]) +
        sign * static_cast<float>(x[row * cols + rotated_col]) * static_cast<float>(sin[idx]));
  }
}

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<float>(a[row * k + kk]) * static_cast<float>(b[kk * n + col]);
  }
  out[row * n + col] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
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
  int batch = blockIdx.x;
  int head = blockIdx.y;
  int dim = threadIdx.x;
  if (batch >= batch_size || head >= num_heads) return;

  int group_size = num_heads / num_kv_heads;
  int kv_head = head / group_size;
  int context_len = static_cast<int>(context_lens[batch]);
  const scalar_t* q = query + (batch * num_heads + head) * head_dim;
  __shared__ float shared[kThreads];

  float max_score = -INFINITY;
  float denom = 0.0f;
  float acc = 0.0f;
  for (int token = 0; token < context_len; ++token) {
    int block_id = static_cast<int>(block_tables[batch * max_blocks + token / block_size]);
    int slot = block_id * block_size + token % block_size;
    const scalar_t* key = key_cache + (slot * num_kv_heads + kv_head) * head_dim;
    const scalar_t* value = value_cache + (slot * num_kv_heads + kv_head) * head_dim;

    float partial = dim < head_dim ? static_cast<float>(q[dim]) * static_cast<float>(key[dim]) : 0.0f;
    shared[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
      __syncthreads();
    }

    float score = shared[0] * scale;
    float next_max = fmaxf(max_score, score);
    float old_scale = expf(max_score - next_max);
    float prob = expf(score - next_max);
    if (dim < head_dim) {
      acc = acc * old_scale + prob * static_cast<float>(value[dim]);
    }
    denom = denom * old_scale + prob;
    max_score = next_max;
    __syncthreads();
  }
  if (dim < head_dim) {
    out[(batch * num_heads + head) * head_dim + dim] = static_cast<scalar_t>(acc / denom);
  }
}

void check_cuda(torch::Tensor tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor softmax(torch::Tensor x) {
  check_cuda(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  auto out = torch::empty_like(x);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "softmax", [&] {
    softmax_kernel<scalar_t><<<x.size(0), kThreads>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), x.size(0), x.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
  check_cuda(x, "x");
  check_cuda(weight, "weight");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "weight must have shape [hidden_size]");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x dtype");
  auto out = torch::empty_like(x);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "rms_norm", [&] {
    rms_norm_kernel<scalar_t><<<x.size(0), kThreads>>>(
        x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), x.size(0), x.size(1), static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor rope(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
  check_cuda(x, "x");
  check_cuda(cos, "cos");
  check_cuda(sin, "sin");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(cos.sizes() == x.sizes() && sin.sizes() == x.sizes(), "cos/sin must match x shape");
  TORCH_CHECK(cos.scalar_type() == x.scalar_type() && sin.scalar_type() == x.scalar_type(), "cos/sin dtype must match x dtype");
  TORCH_CHECK(x.size(1) % 2 == 0, "RoPE head dimension must be even");
  auto out = torch::empty_like(x);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "rope", [&] {
    rope_kernel<scalar_t><<<x.size(0), kThreads>>>(
        x.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(), sin.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), x.size(0), x.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  check_cuda(a, "a");
  check_cuda(b, "b");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul inputs must be 2D");
  TORCH_CHECK(a.size(1) == b.size(0), "matmul shape mismatch");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "matmul input dtypes must match");
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  dim3 block(16, 16);
  dim3 grid((b.size(1) + block.x - 1) / block.x, (a.size(0) + block.y - 1) / block.y);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "matmul", [&] {
    matmul_kernel<scalar_t><<<grid, block>>>(
        a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), a.size(0), b.size(1), a.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor paged_attention_decode(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int64_t block_size,
    double scale) {
  check_cuda(query, "query");
  check_cuda(key_cache, "key_cache");
  check_cuda(value_cache, "value_cache");
  TORCH_CHECK(block_tables.is_cuda() && block_tables.dtype() == torch::kInt64 && block_tables.is_contiguous(), "block_tables must be contiguous CUDA int64");
  TORCH_CHECK(context_lens.is_cuda() && context_lens.dtype() == torch::kInt64 && context_lens.is_contiguous(), "context_lens must be contiguous CUDA int64");
  TORCH_CHECK(query.dim() == 3 && key_cache.dim() == 3 && value_cache.sizes() == key_cache.sizes(), "bad paged attention shapes");
  TORCH_CHECK(query.size(2) == key_cache.size(2), "query and KV head_dim must match");
  TORCH_CHECK(query.size(1) % key_cache.size(1) == 0, "query heads must be divisible by KV heads");
  TORCH_CHECK(query.scalar_type() == key_cache.scalar_type() && query.scalar_type() == value_cache.scalar_type(), "query/KV dtypes must match");
  TORCH_CHECK(query.size(2) <= kThreads, "head_dim must be <= 256 for this teaching kernel");
  auto out = torch::empty_like(query);
  dim3 grid(query.size(0), query.size(1));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "paged_attention_decode", [&] {
    paged_attention_decode_kernel<scalar_t><<<grid, kThreads>>>(
        query.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        block_tables.data_ptr<int64_t>(),
        context_lens.data_ptr<int64_t>(),
        out.data_ptr<scalar_t>(),
        query.size(0),
        query.size(1),
        key_cache.size(1),
        query.size(2),
        block_tables.size(1),
        static_cast<int>(block_size),
        static_cast<float>(scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &softmax, "row-wise softmax");
  m.def("rms_norm", &rms_norm, "RMSNorm");
  m.def("rope", &rope, "Qwen-style RoPE");
  m.def("matmul", &matmul, "naive matrix multiplication");
  m.def("paged_attention_decode", &paged_attention_decode, "paged attention decode");
}
