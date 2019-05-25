#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>

#define NUM_THREADS 256
#include "gaussian_error_kernel.h"

__global__ void gaussian_error_compute_kernel(const float* mean, const float* std_log, const float* values, float* output, const int size){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
	
	float diff = (mean[i]-values[i]) * exp(-std_log[i]);
	float output_val = 0.5f*diff*diff + std_log[i];
	
	if( i < size)
		output[i] = output_val;
	
}

void gaussian_error_compute(const cudaStream_t& stream, const float* mean, const float* std_log, const float* values, float* output, const int size){
	int num_blocks = ((size+NUM_THREADS-1)/NUM_THREADS);
	gaussian_error_compute_kernel<<<num_blocks, NUM_THREADS, 0, stream>>>(mean, std_log, values, output, size);
}



__global__ void gaussian_error_grad_compute_kernel(const float* grad, const float* mean, const float* std_log, const float* values, float* grad_mean, float* grad_std_log, float* grad_values, const int size){
    int i = blockIdx.x * NUM_THREADS + threadIdx.x;
	
	float diff = (mean[i]-values[i]);
	float isig = exp(-std_log[i]);
	
	float grad_mean_val = grad[i]*diff*isig*isig;
	float grad_values_val = -grad_mean_val;
	float grad_std_log_val = grad[i]*(1.0f - diff*diff*isig*isig);
	
	if( i < size){
		grad_mean[i] = grad_mean_val;
		grad_std_log[i] = grad_std_log_val;
		grad_values[i] = grad_values_val;
	}
}

void gaussian_error_grad_compute(const cudaStream_t& stream, const float* grad, const float* mean, const float* std_log, const float* values, float* grad_mean, float* grad_std_log, float* grad_values, const int size){
	int num_blocks = ((size+NUM_THREADS-1)/NUM_THREADS);
	gaussian_error_grad_compute_kernel<<<num_blocks, NUM_THREADS, 0, stream>>>(grad, mean, std_log, values, grad_mean, grad_std_log, grad_values, size);
}