#pragma once

#include <torch/torch.h>
#include <vector>

void trilinear_devoxelize(int b, int c, int n, int r, int r2, int r3,
                          bool is_training, const float *coords,
                          const float *feat, int *inds, float *wgts,
                          float *outs);
void trilinear_devoxelize_grad(int b, int c, int n, int r3, const int *inds,
                               const float *wgts, const float *grad_y,
                               float *grad_x);

std::vector<at::Tensor> trilinear_devoxelize_forward(const int r,
                                                     const bool is_training,
                                                     const at::Tensor coords,
                                                     const at::Tensor features);

at::Tensor trilinear_devoxelize_backward(const at::Tensor grad_y,
                                         const at::Tensor indices,
                                         const at::Tensor weights, const int r);