##################################
# Set path to dependencies.
# CUDA.
CUDA_ROOT=/u/nitish/local/cuda-6.5
###################################

CUDA_LIB=$(CUDA_ROOT)/lib64
CUDA_BIN=$(CUDA_ROOT)/bin

NVCC = $(CUDA_BIN)/nvcc
FLAGS = -O3 --use_fast_math -v \
		    -gencode=arch=compute_20,code=sm_20 \
		    -gencode=arch=compute_30,code=sm_30 \
		    -gencode=arch=compute_35,code=sm_35 \
				--compiler-options '-fPIC' --shared -Xlinker -rpath -Xlinker $(CUDA_LIB) \
				-Xcompiler -rdynamic -lineinfo

all : libcudamat.so

libcudamat.so: cudamat.cu cudamat_kernels.cu cudamat.cuh cudamat_kernels.cuh
	$(NVCC) $(FLAGS) -o $@ cudamat.cu cudamat_kernels.cu -lcublas -L$(CUDA_LIB)

clean:
	rm -rf libcudamat.so
