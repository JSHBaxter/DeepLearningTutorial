#makefile

PYTHON=python3
CC=g++
NVCC=nvcc

CUDAFLAG = -DGOOGLE_CUDA
CUDA_I = -I/usr/local/cuda/include
CUDA_L = /usr/local/cuda-10.0/lib64/libcudart.so.10.0 /usr/local/cuda/lib64/libcublas.so.10.0

TF_CFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; [print(i) for i in tf.sysconfig.get_compile_flags()]') 
TF_LFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; [print(i) for i in tf.sysconfig.get_link_flags()]')

CFLAGS := -std=c++11 $(CUDA_I) $(TF_CFLAGS) $(CUDAFLAG) -fPIC -shared 
CUFLAGS = -std=c++11 -c $(TF_CFLAGS) $(CUDAFLAG) --expt-relaxed-constexpr -x cu -Xcompiler -fPIC -ccbin gcc-7 -DNDEBUG
LFLAGS := $(CUDA_L) $(TF_LFLAGS) 
gaussian_error_kernel.cu.o :
	$(NVCC) $(CUFLAGS) -o gaussian_error_kernel.cu.o gaussian_error_kernel.cu 

gaussian_error.so : gaussian_error_kernel.cu.o gaussian_error.cpp gaussian_error_grad.cpp
	$(CC) $(CFLAGS) -o gaussian_error.so gaussian_error_kernel.cu.o gaussian_error.cpp gaussian_error_grad.cpp $(LFLAGS)

clean :
	rm -rf *.o *.so

all : gaussian_error.so