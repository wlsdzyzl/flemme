#pragma once

#include <torch/torch.h>
#include <vector>

// CUDA function declarations
void avg_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const float *feat, int *ind, int *cnt, float *out);
void avg_voxelize_grad(int b, int c, int n, int s, const int *idx,
                       const int *cnt, const float *grad_y, float *grad_x);

std::vector<at::Tensor> avg_voxelize_forward(const at::Tensor features,
                                             const at::Tensor coords,
                                             const int resolution);

at::Tensor avg_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor cnt);