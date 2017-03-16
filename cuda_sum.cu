#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <helper_cuda.h>


/*extern "C" {
void cuda_float4_array(float4 *a) {
	printf(a[0]);
}

}*/

/////////////////////////////

__global__ void cuda_sum_kernel(float *a, float *b, float *c, size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}

	printf("f");
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
							  //float *x,
							  //float4 *y,
							  float4 *out,
							  size_t n)
   {
   	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx >= n) return;

	printf("%i %i\n", idx, n);
  	//out[idx].x = out[idx].y = out[idx].z = out[idx].w = 0.25; //tex1D<float4>(tex, 0.5);
}

__global__
void hello_world(cudaTextureObject_t tex, float4 *value) {
	printf("\nhi kernel\n");
	value[0].x = 15.0;
	value[0] = tex2D<float4>(tex, 5., 0.5);
	printf("float4 %f %f %f %f", value[0].x, value[0].y, value[0].z, value[0].w);
}

__device__ size_t flatten_2d_index(size_t x, size_t y, size_t w) {
	// write me
	return (y * w) + x;
}

__global__
void resize_kernel(cudaTextureObject_t tex, float4 *output, size_t outw, size_t outh) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= outw || idy >= outh) return;

	// interpolation coordinates (assumes normalized texture coords!!)
	float int_x = idx * (1.0f / float(outw-1));
	float int_y = idy * (1.0f / float(outh-1));

	size_t out_idx = flatten_2d_index(idx, idy, outw);
	//float4 tmp; 
	//tmp.x = tmp.y = tmp.z = tmp.w = 0.5;
	output[out_idx] = tex2D<float4>(tex, int_x, int_y);
}

extern "C" {
void cuda_resize(float4 *image, float4 *new_image, size_t sizew, size_t sizeh, size_t neww, size_t newh)
{
	//size_t n = sizew * sizeh;
	//printf("starting %f", a[0].x);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, sizew, sizeh);
	cudaMemcpyToArray(cuArray, 0, 0, image, sizew*sizeh, cudaMemcpyHostToDevice);
	
	//printf("making res desc");
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	//printf("makign tex desc");
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeClamp;
	texDesc.addressMode[1]   = cudaAddressModeClamp;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;
	//texDesc.normalizedCoords = 0;

	/*cudaResourceViewDesc resViewDesc;
	memset(&resViewDesc, 0, sizeof(resViewDesc));
	resViewDesc.format = cudaResViewFormatFloat1;
	resViewDesc.width = sizew;
	*/

	// create texture object
	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	// make c and interp grid dev pointers
	float4 *d_new_image;
	//float *d_grid;
	checkCudaErrors(cudaMalloc((void **)&d_new_image, neww * newh * sizeof(float4)));
	//cudaMalloc((void **)&d_grid, num_interps * sizeof(float4));
	//cudaMemcpy(d_grid, grid, num_interps * sizeof(float4), cudaMemcpyHostToDevice);

	//printf("launching kernel");
	//int num_interps = 1024;
	//cuda_texture_interpolate<<<ceil(num_interps / 256), 256>>>(tex, d_out, num_interps);



	/*
	float4 *dev;
	cudaMalloc(&dev, sizeof(float4));
	hello_world<<<1,1>>>(tex, dev);

	float4 host;
	cudaMemcpy(&host, dev, sizeof(float4), cudaMemcpyDeviceToHost);
	cudaFree(dev);1d index to 2d
	printf("%f\n", host.x);
	*/
	//cudaFree(d_out);

	dim3 blocksize(8, 8);
	dim3 gridsize(neww / blocksize.x, newh / blocksize.y);

	resize_kernel<<<gridsize, blocksize>>>(tex, d_new_image, neww, newh);

	// copy c back to host
	cudaMemcpy(new_image, d_new_image, neww * newh * sizeof(float4), cudaMemcpyDeviceToHost);

	// clean up
	cudaFree(d_new_image);

	cudaDestroyTextureObject(tex);
	cudaFreeArray(cuArray);
}

}
