#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void cuda_sum_kernel(float *a, float *b, float *c, size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}

	c[idx] = a[idx] + b[idx];
}

extern "C" {
void cuda_sum(float *a, float *b, float *c, size_t size)
{
	float *d_a, *d_b, *d_c;

	cudaMalloc((void **)&d_a, size * sizeof(float));
	cudaMalloc((void **)&d_b, size * sizeof(float));
	cudaMalloc((void **)&d_c, size * sizeof(float));

	cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

	cuda_sum_kernel <<< ceil(size / 256.0), 256 >>> (d_a, d_b, d_c, size);

	cudaMemcpy(c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}
}

// attempt to interpolate linear memory
__global__
void cuda_texture_interpolate(cudaTextureObject_t tex,
							  float *x,
							  float *y,
							  int n)
   {
   	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx > n) return;

  	y[idx] = tex1D<float>(tex, x[idx]);

/*
  if (count < 1) { count = 1; }
  float h = (stop-start)/((float)count);
  float x = start;
  float y;
  for (int i = 0; i != count; i++) {
	y = tex1D<float>(tex,x);
	printf("x: %4g ; y: %4g\n",x,y);
	x = x + h;
  }
  y = tex1D<float>(tex,x);
  printf("x: %4g ; y: %4g\n",x,y);*/
}

extern "C" {
void cuda_interp1D(float *a, float *c, float *grid, size_t size, size_t num_interps)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, size);
	cudaMemcpyToArray(cuArray, 0, 0, a, size*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeClamp;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;
	//texDesc.normalizedCoords = 0;

	cudaResourceViewDesc resViewDesc;
	memset(&resViewDesc, 0, sizeof(resViewDesc));
	resViewDesc.format = cudaResViewFormatFloat1;
	resViewDesc.width = size;

	// create texture object
	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, &resViewDesc);

	// make c and interp grid dev pointers
	float *d_c;
	float *d_grid;
	cudaMalloc((void **)&d_c, num_interps * sizeof(float));
	cudaMalloc((void **)&d_grid, num_interps * sizeof(float));
	cudaMemcpy(d_grid, grid, num_interps * sizeof(float), cudaMemcpyHostToDevice);

	cuda_texture_interpolate<<<ceil(num_interps / 256), 256>>>(tex, d_grid, d_c, num_interps);

	 // copy c back to host
	cudaMemcpy(c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

	// clean up
	cudaFree(d_grid);
	cudaFree(d_c);

	cudaDestroyTextureObject(tex);
	cudaFreeArray(cuArray);
}

}