#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"
#include "voxelize.h"
#include "trilinear_devox.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_points", &gather_points);
  m.def("gather_points_grad", &gather_points_grad);
  m.def("furthest_point_sampling", &furthest_point_sampling);

  m.def("three_nn", &three_nn);
  m.def("three_interpolate", &three_interpolate);
  m.def("three_interpolate_grad", &three_interpolate_grad);

  m.def("ball_query", &ball_query);
  m.def("batch_ball_query", &batch_ball_query);
  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);

  m.def("trilinear_devoxelize_forward", &trilinear_devoxelize_forward,
        "Trilinear Devoxelization forward (CUDA)");
  m.def("trilinear_devoxelize_backward", &trilinear_devoxelize_backward,
        "Trilinear Devoxelization backward (CUDA)");
  m.def("avg_voxelize_forward", &avg_voxelize_forward,
        "Voxelization forward with average pooling (CUDA)");
  m.def("avg_voxelize_backward", &avg_voxelize_backward,
        "Voxelization backward (CUDA)");
}
