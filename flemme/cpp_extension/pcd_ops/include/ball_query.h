#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);
std::vector<at::Tensor> batch_ball_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor radius,
                      const int nsample);