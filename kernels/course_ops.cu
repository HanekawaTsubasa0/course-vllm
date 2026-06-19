#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

namespace {

constexpr int kThreads = 256;

template <typename T>
__device__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ T from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __half from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename scalar_t>
__global__ void softmax_kernel(const scalar_t* x, scalar_t* out, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* row_x = x + row * cols;
  float row_max = -INFINITY;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    row_max = fmaxf(row_max, to_float(row_x[col]));
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
    denom += expf(to_float(row_x[col]) - row_max);
  }
  shared[threadIdx.x] = denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  denom = shared[0];
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[row * cols + col] = from_float<scalar_t>(expf(to_float(row_x[col]) - row_max) / denom);
  }
}

template <typename scalar_t>
__global__ void rms_norm_kernel(const scalar_t* x, const scalar_t* weight, scalar_t* out, int rows, int cols, float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* row_x = x + row * cols;
  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = to_float(row_x[col]);
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
    out[row * cols + col] = from_float<scalar_t>(to_float(row_x[col]) * scale * to_float(weight[col]));
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
    out[idx] = from_float<scalar_t>(to_float(x[idx]) * to_float(cos[idx]) + sign * to_float(x[row * cols + rotated_col]) * to_float(sin[idx]));
  }
}

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += to_float(a[row * k + kk]) * to_float(b[kk * n + col]);
  }
  out[row * n + col] = from_float<scalar_t>(acc);
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

    float partial = dim < head_dim ? to_float(q[dim]) * to_float(key[dim]) : 0.0f;
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
      acc = acc * old_scale + prob * to_float(value[dim]);
    }
    denom = denom * old_scale + prob;
    max_score = next_max;
    __syncthreads();
  }
  if (dim < head_dim) {
    out[(batch * num_heads + head) * head_dim + dim] = from_float<scalar_t>(acc / denom);
  }
}

template <typename scalar_t>
int launch_softmax(const void* x, void* out, int rows, int cols) {
  softmax_kernel<<<rows, kThreads>>>(static_cast<const scalar_t*>(x), static_cast<scalar_t*>(out), rows, cols);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_rms_norm(const void* x, const void* weight, void* out, int rows, int cols, float eps) {
  rms_norm_kernel<<<rows, kThreads>>>(static_cast<const scalar_t*>(x), static_cast<const scalar_t*>(weight), static_cast<scalar_t*>(out), rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_rope(const void* x, const void* cos, const void* sin, void* out, int rows, int cols) {
  rope_kernel<<<rows, kThreads>>>(
      static_cast<const scalar_t*>(x), static_cast<const scalar_t*>(cos), static_cast<const scalar_t*>(sin), static_cast<scalar_t*>(out), rows, cols);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_matmul(const void* a, const void* b, void* out, int m, int n, int k) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  matmul_kernel<<<grid, block>>>(static_cast<const scalar_t*>(a), static_cast<const scalar_t*>(b), static_cast<scalar_t*>(out), m, n, k);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_paged_attention_decode(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const int64_t* block_tables,
    const int64_t* context_lens,
    void* out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks,
    int block_size,
    float scale) {
  dim3 grid(batch_size, num_heads);
  paged_attention_decode_kernel<<<grid, kThreads>>>(
      static_cast<const scalar_t*>(query),
      static_cast<const scalar_t*>(key_cache),
      static_cast<const scalar_t*>(value_cache),
      block_tables,
      context_lens,
      static_cast<scalar_t*>(out),
      batch_size,
      num_heads,
      num_kv_heads,
      head_dim,
      max_blocks,
      block_size,
      scale);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int softmax_cuda_launcher(const void* x, void* out, int rows, int cols, int dtype) {
  if (dtype == 0) return launch_softmax<float>(x, out, rows, cols);
  if (dtype == 1) return launch_softmax<__half>(x, out, rows, cols);
  return launch_softmax<__nv_bfloat16>(x, out, rows, cols);
}

extern "C" int rms_norm_cuda_launcher(const void* x, const void* weight, void* out, int rows, int cols, float eps, int dtype) {
  if (dtype == 0) return launch_rms_norm<float>(x, weight, out, rows, cols, eps);
  if (dtype == 1) return launch_rms_norm<__half>(x, weight, out, rows, cols, eps);
  return launch_rms_norm<__nv_bfloat16>(x, weight, out, rows, cols, eps);
}

extern "C" int rope_cuda_launcher(const void* x, const void* cos, const void* sin, void* out, int rows, int cols, int dtype) {
  if (dtype == 0) return launch_rope<float>(x, cos, sin, out, rows, cols);
  if (dtype == 1) return launch_rope<__half>(x, cos, sin, out, rows, cols);
  return launch_rope<__nv_bfloat16>(x, cos, sin, out, rows, cols);
}

extern "C" int matmul_cuda_launcher(const void* a, const void* b, void* out, int m, int n, int k, int dtype) {
  if (dtype == 0) return launch_matmul<float>(a, b, out, m, n, k);
  if (dtype == 1) return launch_matmul<__half>(a, b, out, m, n, k);
  return launch_matmul<__nv_bfloat16>(a, b, out, m, n, k);
}

extern "C" int paged_attention_decode_cuda_launcher(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const int64_t* block_tables,
    const int64_t* context_lens,
    void* out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks,
    int block_size,
    float scale,
    int dtype) {
  if (dtype == 0) {
    return launch_paged_attention_decode<float>(
        query, key_cache, value_cache, block_tables, context_lens, out, batch_size, num_heads, num_kv_heads, head_dim, max_blocks, block_size, scale);
  }
  if (dtype == 1) {
    return launch_paged_attention_decode<__half>(
        query, key_cache, value_cache, block_tables, context_lens, out, batch_size, num_heads, num_kv_heads, head_dim, max_blocks, block_size, scale);
  }
  return launch_paged_attention_decode<__nv_bfloat16>(
      query, key_cache, value_cache, block_tables, context_lens, out, batch_size, num_heads, num_kv_heads, head_dim, max_blocks, block_size, scale);
}
