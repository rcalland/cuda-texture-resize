#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <helper_cuda.h>

__device__ size_t flatten_2d_index(size_t x, size_t y, size_t w) {
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
	//tmp.x = tmp.y = tmp.z = tmp.w = int_y;
	output[out_idx] = tex2D<float4>(tex, int_x, int_y);
}

extern "C" {
void cuda_resize(float4 *image, float4 *new_image, size_t sizew, size_t sizeh, size_t neww, size_t newh)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, sizew, sizeh);
	cudaMemcpyToArray(cuArray, 0, 0, image, sizew*sizeh*sizeof(float4), cudaMemcpyHostToDevice);
	
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeClamp;
	texDesc.addressMode[1]   = cudaAddressModeClamp;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// create texture object
	cudaTextureObject_t tex = 0;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	// make c and interp grid dev pointers
	float4 *d_new_image;
	checkCudaErrors(cudaMalloc((void **)&d_new_image, neww * newh * sizeof(float4)));

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
