#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"


#ifdef GOOGLE_CUDA 
#define EIGEN_USE_GPU
#endif
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>

#include <math.h>

// Use appropriate namespaces
using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
#ifdef GOOGLE_CUDA 
using GPUDevice = Eigen::GpuDevice;
#endif

template <typename Device>
void compute_gaussian_error_grad(const Device& dev, const float*  grad, const float* means, const float* std_log, const float* values, float* grad_means, float* grad_std_log, float* grad_values, const int size);

// Define the OpKernel class
template <typename Device>
class GaussianErrorGradOp : public OpKernel {
public:
    explicit GaussianErrorGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
		
        // ensure all inputs are present
        DCHECK_EQ(4, context->num_inputs());

        // get the input tensors
        const Tensor* means = &(context->input(0));
        const Tensor* sigma_log = &(context->input(1));
        const Tensor* samples = &(context->input(2));
        const Tensor* grad = &(context->input(3));
		
        // get shapes of inputs
        const DataType data_type = means->dtype();
        const TensorShape& means_shape = means->shape();
        const TensorShape& sigma_log_shape = sigma_log->shape();
        const TensorShape& samples_shape = samples->shape();
        const TensorShape& grad_shape = grad->shape();
		
		// ensure data is the same rank and shape
		int rank = means_shape.dims();
        DCHECK_EQ(sigma_log_shape.dims(), rank);
        DCHECK_EQ(samples_shape.dims(), rank);
        DCHECK_EQ(grad_shape.dims(), rank);
		for(int r = 0; r < rank; r++) {
			DCHECK_EQ(means_shape.dim_size(r), sigma_log_shape.dim_size(r));
			DCHECK_EQ(means_shape.dim_size(r), samples_shape.dim_size(r));
			DCHECK_EQ(means_shape.dim_size(r), grad_shape.dim_size(r));
		}
		
		// get total size of buffer
		int size = 1;
		for(int r = 0; r < rank; r++)
			size *= (int) means_shape.dim_size(r);

        // create output tensor
        Tensor* grad_means = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, means_shape, &grad_means));
        Tensor* grad_sigma_log = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, means_shape, &grad_sigma_log));
        Tensor* grad_samples = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, means_shape, &grad_samples));
		
		// run the computation
		compute_gaussian_error_grad<Device>(
			context->eigen_device<Device>(),
            grad->flat<float>().data(),
            means->flat<float>().data(),
            sigma_log->flat<float>().data(),
            samples->flat<float>().data(),
            grad_means->flat<float>().data(),
            grad_sigma_log->flat<float>().data(),
            grad_samples->flat<float>().data(),
			size );
		
	}
};

// Register operation interface
REGISTER_OP("GaussianErrorGrad")
  .Input("means: float")
  .Input("sigma_log: float")
  .Input("samples: float")
  .Input("grad: float")
  .Output("grad_means: float")
  .Output("grad_sigma_log: float")
  .Output("grad_samples: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    c->set_output(2, c->input(2));
    return Status::OK();
  });

// Register operations as kernels.
REGISTER_KERNEL_BUILDER(Name("GaussianErrorGrad").Device(DEVICE_CPU), GaussianErrorGradOp<CPUDevice>);
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("GaussianErrorGrad").Device(DEVICE_GPU), GaussianErrorGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA



template <>
void compute_gaussian_error_grad<CPUDevice>(const CPUDevice& dev, const float*  grad, const float* means, const float* std_log, const float* values, float* grad_means, float* grad_std_log, float* grad_values, const int size){
	for(int i = 0; i < size; i++){
		float diff = means[i]-values[i];
		float isig = std::exp(-std_log[i]);
		grad_means[i] = grad[i]*diff*isig*isig;
		grad_values[i] = -grad[i]*diff*isig*isig;
		grad_std_log[i] = grad[i]*(1.0f - diff*diff*isig*isig);
	}
}

 
#ifdef GOOGLE_CUDA 
#define EIGEN_USE_GPU
#include "gaussian_error_kernel.h"
template <>
void compute_gaussian_error_grad<GPUDevice>(const GPUDevice& dev, const float*  grad, const float* means, const float* std_log, const float* values, float* grad_means, float* grad_std_log, float* grad_values, const int size){
	gaussian_error_grad_compute(dev.stream(), grad, means, std_log, values, grad_means, grad_std_log, grad_values, size);
}
#endif

