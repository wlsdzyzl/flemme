#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx, 
                                        float *__restrict__ dist) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;
  dist += m * nsample * batch_index;
  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
            dist[j * nsample + l] = d2;
          }
        }
        idx[j * nsample + cnt] = k;
        dist[j * nsample + cnt] = d2; 
        ++cnt;
      }
    }
  }
}


__global__ void batch_query_ball_point_kernel(int b, int n, int m, 
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        const float *__restrict__ radius,
                                        int *__restrict__ idx, 
                                        float *__restrict__ dist) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  radius += batch_index * m;

  idx += m * nsample * batch_index;
  dist += m * nsample * batch_index;
  
  int index = threadIdx.x;
  int stride = blockDim.x;

  
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    float radius2 = radius[j] * radius[j];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
            dist[j * nsample + l] = d2;
          }
        }
        idx[j * nsample + cnt] = k;
        dist[j * nsample + cnt] = d2; 
        ++cnt;
      }
    }
  }
}


// find the nsample nearest point in the ball
__global__ void nearest_query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx, 
                                        float *__restrict__ dist) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;
  dist += m * nsample * batch_index;
  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
            dist[j * nsample + l] = d2;
          }
        }
        idx[j * nsample + cnt] = k;
        dist[j * nsample + cnt] = d2; 
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx, float *dist) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, idx, dist);

  CUDA_CHECK_ERRORS();
}

void batch_query_ball_point_kernel_wrapper(int b, int n, int m, 
                                     int nsample, const float *new_xyz,
                                     const float *xyz, 
                                     const float *radius,
                                     int *idx, float *dist) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  batch_query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, nsample, new_xyz, xyz, radius, idx, dist);

  CUDA_CHECK_ERRORS();
}

