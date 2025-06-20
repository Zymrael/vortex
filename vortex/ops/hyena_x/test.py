import time 

import torch
import torch.nn as nn

from triton_indirect_fwd import (
    full_hyena_x_fwd,
    hyena_x_bdl_fwd,
    hyena_x_bdl_dchunk_fwd,
    hyena_x_fwd_causal_conv1d,
    hyena_x_bdl_dchunk_independent_fwd, 
    hyena_x_fwd_ref,
    HyenaXInnerFunc,
)


torch.manual_seed(1234)
ATOL = 1e-2
RTOL = 1e-2
L = 256
D = 2048
B = 32
L_h = 16


def get_inputs(B, D, L, L_h, input_type="(qkv, h)"):
    dtype = torch.bfloat16
    u = torch.randn((B, L, D), dtype=dtype, device='cuda')
    lin_featurizer = nn.Linear(D, 3 * D, bias=False).to(torch.device('cuda')).to(dtype)
    lin_out_featurizer = nn.Linear(D, D, bias=False).to(torch.device('cuda')).to(dtype)
    q = torch.rand((B, D, L), dtype=dtype, device='cuda') - 0.5
    k = torch.rand((B, D, L), dtype=dtype, device='cuda') - 0.5
    v = torch.rand((B, D, L), dtype=dtype, device='cuda') - 0.5
    h = torch.rand((3 * D, 1, L_h), dtype=dtype, device='cuda') - 0.5
    if input_type == "(qkv, h)":
        qkv = torch.cat([q, k, v], dim=1)
        return qkv, h
    elif input_type == "(q, k, v, h)":
        return q, k, v, h
    elif input_type == "(u, h, lin_featurizer, lin_out_featurizer)":
        return u, h, lin_featurizer, lin_out_featurizer


qkv, h = get_inputs(B, D, L, L_h, input_type="(qkv, h)")
q, k, v = qkv.split([D, D, D], dim=1)
q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

print(f"Running HyenaXInnerFunc...")
func = HyenaXInnerFunc.apply
y = func(q, k, v, h)
print(f"HyenaXInnerFunc complete: {y.shape}", end='\n\n')

print(f"Running reference Hyena-X forward pass...")
y_ref = hyena_x_fwd_ref(qkv=qkv, h=h)
print(f"Reference Hyena-X forward pass complete: {y_ref.shape}", end='\n\n')

print(f"Running BDL Hyena-X forward pass...")
y_bdl_kernel = hyena_x_bdl_fwd(q=q, k=k, v=v, h=h)
print(f"BDL Hyena-X forward pass complete: {y_bdl_kernel.shape}", end='\n\n')

print(f"Running BDL Independent Hyena-X forward pass...")
y_bdl_independent_kernel = hyena_x_bdl_dchunk_independent_fwd(q=q, k=k, v=v, h=h)
print(f"BDL Independent Hyena-X forward pass complete: {y_bdl_independent_kernel.shape}", end='\n\n')

print(y_bdl_independent_kernel, y_ref)
print(f"Testing accuracy with atol={ATOL}, rtol={RTOL}...")
torch.testing.assert_close(y_ref, y_bdl_kernel, atol=ATOL, rtol=RTOL)
#torch.testing.assert_close(y_ref, y_bdl_independent_kernel, atol=ATOL, rtol=RTOL)
print(f"Accuracy test complete", end='\n\n')


def benchmark_impl(func, input_type="(qkv, h)", warmup=10, repeat=10):
    inputs = get_inputs(B, D, L, L_h, input_type)
    for _ in range(warmup):
        y = func(*inputs)
        torch.cuda.synchronize()
    
    total_time = 0
    for _ in range(repeat):
        inputs = get_inputs(B, D, L, L_h, input_type)

        start = time.time()
        with torch.inference_mode():
            y = func(*inputs)
        torch.cuda.synchronize()
        end = time.time() 
        total_time += (end - start)
    return total_time / repeat


for func in [hyena_x_fwd_ref, hyena_x_fwd_causal_conv1d]:
    if func.__name__ == "hyena_x_fwd_causal_conv1d" and L_h > 4:
        break
    print(f"Benchmarking {func.__name__}...")
    t = benchmark_impl(func=func, input_type="(qkv, h)", repeat=20, warmup=5)
    print(f"{func.__name__} complete: {t * 1e6} microseconds", end='\n\n')

for func in [hyena_x_bdl_dchunk_fwd]:
    print(f"Benchmarking {func.__name__}...")
    t = benchmark_impl(func=func, input_type="(q, k, v, h)", repeat=20, warmup=5)
    print(f"{func.__name__} complete: {t * 1e6} microseconds", end='\n\n')

# print(f"Benchmarking full Hyena-X forward pass...")
# t = benchmark_impl(func=full_hyena_x_fwd, input_type="(u, h, lin_featurizer, lin_out_featurizer)", repeat=20, warmup=5)
# print(f"Full Hyena-X forward pass complete: {t * 1e6} microseconds", end='\n\n')
