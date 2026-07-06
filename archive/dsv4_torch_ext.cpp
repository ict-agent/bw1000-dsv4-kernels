// dsv4_torch_ext.cpp - PyTorch C++ extension wrapper for HIP kernels
// Proper stream handling + no ctypes overhead
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

extern "C" {
    void launch_ptq(const void* x, void* q, void* s, int M, int N, hipStream_t st);
    void launch_silu_mul(const void* g, const void* u, void* out, int M, int N, hipStream_t st);
    void launch_silu_mul_quant(const void* g, const void* u, void* q, void* s, int M, int N, hipStream_t st);
    void launch_rmsnorm(const void* x, void* out, int M, int N, float eps, hipStream_t st);
    void launch_add_rmsnorm_quant(void* r, const void* x, const void* w, void* q, void* s, int M, int N, float eps, hipStream_t st);
}

static inline hipStream_t get_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

std::vector<torch::Tensor> per_token_quant_int8_hip(torch::Tensor x) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    int M = (int)(x.numel() / x.size(-1));
    int N = (int)x.size(-1);
    auto q = torch::empty_like(x, x.options().dtype(torch::kInt8));
    auto s = torch::empty({M}, x.options().dtype(torch::kFloat32));
    launch_ptq(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, get_stream());
    return {q, s.unsqueeze(-1)};
}

torch::Tensor silu_and_mul_hip(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "inputs must be contiguous");
    int M = (int)gate.size(0);
    int N = (int)gate.size(1);
    auto out = torch::empty_like(gate);
    launch_silu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(), M, N, get_stream());
    return out;
}

std::vector<torch::Tensor> silu_mul_quant_hip(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "inputs must be contiguous");
    int M = (int)gate.size(0);
    int N = (int)gate.size(1);
    auto q = torch::empty_like(gate, gate.options().dtype(torch::kInt8));
    auto s = torch::empty({M}, gate.options().dtype(torch::kFloat32));
    launch_silu_mul_quant(gate.data_ptr(), up.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, get_stream());
    return {q, s.unsqueeze(-1)};
}

torch::Tensor rmsnorm_hip(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    int M = (int)x.size(0);
    int N = (int)x.size(1);
    auto out = torch::empty_like(x);
    launch_rmsnorm(x.data_ptr(), out.data_ptr(), M, N, (float)eps, get_stream());
    return out;
}

std::vector<torch::Tensor> add_rmsnorm_quant_hip(torch::Tensor residual, torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    int M = (int)x.size(0);
    int N = (int)x.size(1);
    auto q = torch::empty_like(x, x.options().dtype(torch::kInt8));
    auto s = torch::empty({M}, x.options().dtype(torch::kFloat32));
    launch_add_rmsnorm_quant(residual.data_ptr(), x.data_ptr(), weight.data_ptr(),
                             q.data_ptr(), s.data_ptr(), M, N, (float)eps, get_stream());
    return {q, s.unsqueeze(-1)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("per_token_quant_int8", &per_token_quant_int8_hip);
    m.def("silu_and_mul", &silu_and_mul_hip);
    m.def("silu_mul_quant", &silu_mul_quant_hip);
    m.def("rmsnorm", &rmsnorm_hip);
    m.def("add_rmsnorm_quant", &add_rmsnorm_quant_hip);
}
