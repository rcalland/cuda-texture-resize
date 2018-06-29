nvcc -O3 -Xcompiler -fPIC -shared -I /usr/local/cuda/samples/common/inc/ -o cuda_resize.so cuda_resize.cu
