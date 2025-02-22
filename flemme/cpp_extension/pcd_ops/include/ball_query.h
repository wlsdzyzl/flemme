#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);
