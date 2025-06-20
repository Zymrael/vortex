#### Custom kernels for Hyena-X 

* Forward: a Triton indirect convolution kernel for depthwise FIR convolutions over feature groups `q`, `k`, `v`, followed by their elementwise gating. 
* Forward: a reference implementation using PyTorch's native conv1d backends. 
* Forward: a reference implementation using a custom `causal_conv1d` CUDA kernel for small kernel sizes (2, 4). 


#### Notes

Left to do: 
* Improving numerical accuracy
* Figuring reason behind slow elemwise ops (use inline PTX)
* TMA version







