#include "ball_query.h"
#include "utils.h"
#include <vector>

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx, float *dist);

void batch_query_ball_point_kernel_wrapper(int b, int n, int m, 
                                     int nsample, const float *new_xyz,
                                     const float *xyz, 
                                     const float *radius,
                                     int *idx, float *dist);
std::vector<at::Tensor> ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      -torch::ones({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));
  at::Tensor dist =
      -torch::ones({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Float));
  if (new_xyz.is_cuda()) {
    query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data_ptr<float>(),
                                    xyz.data_ptr<float>(), idx.data_ptr<int>(), 
                                    dist.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return std::vector<at::Tensor>{idx, dist};
}

std::vector<at::Tensor> batch_ball_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(radius);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(radius);

  if (new_xyz.is_cuda()) {
    CHECK_CUDA(xyz);
    CHECK_CUDA(radius);
  }

  at::Tensor idx =
      -torch::ones({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));
  at::Tensor dist =
      -torch::ones({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Float));
  if (new_xyz.is_cuda()) {
    batch_query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    nsample, new_xyz.data_ptr<float>(),
                                    xyz.data_ptr<float>(), 
                                    radius.data_ptr<float>(),
                                    idx.data_ptr<int>(), 
                                    dist.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return std::vector<at::Tensor>{idx, dist};
}