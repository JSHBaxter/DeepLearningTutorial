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
void compute_gaussian_error(const Device& dev, float*  output_error, const float* means, const float* std_log, const float* values, float* diff, const int size);

// Define the OpKernel class
template <typename Device>
class GaussianErrorOp : public OpKernel {
public:
    explicit GaussianErrorOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
		
        // ensure all inputs are present
        DCHECK_EQ(3, context->num_inputs());

        // get the input tensors
        const Tensor* means = &(context->input(0));
        const Tensor* sigma_log = &(context->input(1));
        const Tensor* samples = &(context->input(2));
		
        // get shapes of inputs
        const DataType data_type = means->dtype();
        const TensorShape& means_shape = means->shape();
        const TensorShape& sigma_log_shape = sigma_log->shape();
        const TensorShape& samples_shape = samples->shape();
		
		// ensure data is the same rank and shape
		int rank = means_shape.dims();
        DCHECK_EQ(sigma_log_shape.dims(), rank);
        DCHECK_EQ(samples_shape.dims(), rank);
		for(int r = 0; r < rank; r++) {
			DCHECK_EQ(means_shape.dim_size(r), sigma_log_shape.dim_size(r));
			DCHECK_EQ(means_shape.dim_size(r), samples_shape.dim_size(r));
		}
		
		// get total size of buffer
		int size = 1;
		for(int r = 0; r < rank; r++)
			size *= (int) means_shape.dim_size(r);

        // create output tensor
        Tensor* output_error = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, means_shape, &output_error));
		
		// create temporary buffer
		Tensor diff;
		OP_REQUIRES_OK(context, context->allocate_temp(means->dtype(), means_shape, &diff));
		
		// run the computation
		compute_gaussian_error<Device>(
			context->eigen_device<Device>(),
            output_error->flat<float>().data(),
            means->flat<float>().data(),
            sigma_log->flat<float>().data(),
            samples->flat<float>().data(),
			diff.flat<float>().data(),
			size );
		
	}
};

// Register operation interface
REGISTER_OP("GaussianError")
  .Input("means: float")
  .Input("sigma_log: float")
  .Input("samples: float")
  .Output("output_error: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input;
    c->set_output(0, c->input(0));
    return Status::OK();
  });

// Register operations as kernels.
REGISTER_KERNEL_BUILDER(Name("GaussianError").Device(DEVICE_CPU), GaussianErrorOp<CPUDevice>);
#ifdef GOOGLE_CUDA 
REGISTER_KERNEL_BUILDER(Name("GaussianError").Device(DEVICE_GPU), GaussianErrorOp<GPUDevice>);
#endif  // GOOGLE_CUDA



template <>
void compute_gaussian_error<CPUDevice>(const CPUDevice& dev, float*  output_error, const float* means, const float* std_log, const float* values, float* diff, const int size){
	for(int i = 0; i < size; i++){
		diff[i] = (means[i]-values[i])*std::exp(-std_log[i]);
		output_error[i] = 0.5f*diff[i]*diff[i] + std_log[i];
	}
}


#ifdef GOOGLE_CUDA 
#define EIGEN_USE_GPU
#include "gaussian_error_kernel.h"
template <>
void compute_gaussian_error<GPUDevice>(const GPUDevice& dev, float*  output_error, const float* means, const float* std_log, const float* values, float* diff, const int size){
	gaussian_error_compute(dev.stream(),means,std_log,values,output_error,size);
}
#endif

