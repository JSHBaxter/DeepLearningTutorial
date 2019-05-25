#makefile

CUDAFLAG = -DGOOGLE_CUDA
CUDA_I = /usr/local/cuda-10.0/lib64/libcudart.so.10.0 /usr/local/cuda/lib64/libcublas.so.10.0

MAKEFFL:=$(shell echo "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))" > flags_l.tmp)
MAKEFFC:=$(shell echo "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))" > flags_c.tmp)
TF_LFLAGS:=$(shell python3 flags_l.tmp)
TF_CFLAGS:=$(shell python3 flags_c.tmp) -I/usr/local/cuda/include

gaussian_error_kernel.cu.o :
	nvcc -std=c++11 -c -o gaussian_error_kernel.cu.o gaussian_error_kernel.cu $(TF_CFLAGS) $(CUDAFLAG) --expt-relaxed-constexpr -x cu -Xcompiler -fPIC -ccbin gcc-7 -DNDEBUG


gaussian_error.so : gaussian_error_kernel.cu.o gaussian_error.cpp gaussian_error_grad.cpp
	g++ -std=c++11 -shared -o gaussian_error.so gaussian_error_kernel.cu.o gaussian_error.cpp gaussian_error_grad.cpp $(CUDA_I) $(TF_CFLAGS) $(CUDAFLAG) -fPIC $(TF_LFLAGS)
	rm flags_l.tmp flags_c.tmp

all : gaussian_error.so

clean :
	rm -rf *.o *.so