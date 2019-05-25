#ifndef GAUSSIAN_ERROR_H
#define GAUSSIAN_ERROR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op_kernel.h"


void gaussian_error_compute(const cudaStream_t& stream, const float* mean, const float* std_log, const float* values, float* output, const int size);
void gaussian_error_grad_compute(const cudaStream_t& stream, const float* grad, const float* mean, const float* std_log, const float* values, float* grad_mean, float* grad_std_log, float* grad_values, const int size);

#endif