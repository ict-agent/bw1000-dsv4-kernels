"""
PyTorch extension wrapper for HIP fused kernels.
Compiles and loads the HIP kernels, provides Python API for benchmarking and engine integration.
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

# HIP kernel source
HIP_SOURCE = open(os.path.join(os.path.dirname(__file__), "fused_ops.hip")).read()

# PyTorch C++ wrapper
CPP_SOURCE = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Declarations from fused_ops.hip
extern "C" void launch_fused_add_rmsnorm_quant(
    void* residual, const void* x, const void* weight,
    void* out_int8, void* out_scale,
    int M, int N, float eps, hipStream_t stream);

extern "C" void launch_fused_silu_mul_quant(
    const void* gate, const void* up,
    void* out_int8, void* out_scale,
    int M, int N, hipStream_t stream);

extern "C" void launch_fused_rmsnorm_quant(
    const void* x, const void* weight,
    void* out_int8, void* out_scale,
    int M, int N, float eps, hipStream_t stream);

std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm_quant(
    torch::Tensor residual, torch::Tensor x, torch::Tensor weight, float eps) {
    int M = residual.size(0);
    int N = residual.size(1);
    auto out_int8 = torch::empty({M, N}, torch::dtype(torch::kInt8).device(residual.device()));
    auto out_scale = torch::empty({M}, torch::dtype(torch::kFloat32).device(residual.device()));
    auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    launch_fused_add_rmsnorm_quant(
        residual.data_ptr(), x.data_ptr(), weight.data_ptr(),
        out_int8.data_ptr(), out_scale.data_ptr(),
        M, N, eps, stream);
    return std::make_tuple(out_int8, out_scale);
}

std::tuple<torch::Tensor, torch::Tensor> fused_silu_mul_quant(
    torch::Tensor gate, torch::Tensor up) {
    int M = gate.size(0);
    int N = gate.size(1);
    auto out_int8 = torch::empty({M, N}, torch::dtype(torch::kInt8).device(gate.device()));
    auto out_scale = torch::empty({M}, torch::dtype(torch::kFloat32).device(gate.device()));
    auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    launch_fused_silu_mul_quant(
        gate.data_ptr(), up.data_ptr(),
        out_int8.data_ptr(), out_scale.data_ptr(),
        M, N, stream);
    return std::make_tuple(out_int8, out_scale);
}

std::tuple<torch::Tensor, torch::Tensor> fused_rmsnorm_quant(
    torch::Tensor x, torch::Tensor weight, float eps) {
    int M = x.size(0);
    int N = x.size(1);
    auto out_int8 = torch::empty({M, N}, torch::dtype(torch::kInt8).device(x.device()));
    auto out_scale = torch::empty({M}, torch::dtype(torch::kFloat32).device(x.device()));
    auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    launch_fused_rmsnorm_quant(
        x.data_ptr(), weight.data_ptr(),
        out_int8.data_ptr(), out_scale.data_ptr(),
        M, N, eps, stream);
    return std::make_tuple(out_int8, out_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_rmsnorm_quant", &fused_add_rmsnorm_quant,
          "Fused Add + RMSNorm + INT8 Quantization");
    m.def("fused_silu_mul_quant", &fused_silu_mul_quant,
          "Fused SiLU * Mul + INT8 Quantization");
    m.def("fused_rmsnorm_quant", &fused_rmsnorm_quant,
          "Fused RMSNorm + INT8 Quantization");
}
"""


def build_extension():
    """Build the HIP extension using torch cpp_extension."""
    hip_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fused_ops.hip")
    module = load_inline(
        name="fused_hip_ops",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[HIP_SOURCE],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx936"],
        verbose=True,
    )
    return module


if __name__ == "__main__":
    print("Building HIP fused ops extension...")
    mod = build_extension()
    print(f"Module loaded: {mod}")
    print(f"Functions: {[x for x in dir(mod) if not x.startswith('_')]}")
