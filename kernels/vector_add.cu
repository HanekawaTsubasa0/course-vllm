#include <torch/extension.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = a[idx] + b[idx];
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
