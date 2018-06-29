#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <helper_cuda.h>

__device__ __host__ __inline__ size_t flatten_2d_index(const size_t x, const size_t y, const size_t w) {
	return (y * w) + x;
}

__global__
void resize_kernel(const cudaTextureObject_t tex, float4 __restrict__ *output, const float outw, const float outh) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= outw || idy >= outh) return;

	// interpolation coordinates (assumes normalized texture coords!!)
	float int_x = idx * outw;
	float int_y = idy * outh;

	size_t out_idx = flatten_2d_index(idx, idy, outw);
	//float4 tmp; 
	//tmp.x = tmp.y = tmp.z = tmp.w = int_y;
	output[out_idx] = tex2D<float4>(tex, int_x, int_y);
}

extern "C" {
void cuda_resize(const float4 *image, float4 *new_image, const size_t sizew, const size_t sizeh, const size_t neww, const size_t newh)
{
	
	//
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	//cudaArray* cuArray;
	//cudaMallocArray(&cuArray, &channelDesc, sizew, sizeh);
	//cudaMemcpyToArray(cuArray, 0, 0, image, sizew*sizeh*sizeof(float4), cudaMemcpyHostToDevice);
	//

	//
	size_t pitch;
	float4 *d_image = 0;
    cudaMallocPitch<float4>(&d_image, &pitch, sizew*sizeof(float4), sizeh);
    cudaMemcpy2D(d_image, pitch, image, sizew*sizeof(float4),
                            sizew*sizeof(float4), sizeh, cudaMemcpyHostToDevice);
	//

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	//resDesc.resType = cudaResourceTypeArray;
	//resDesc.res.array.array = cuArray;
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float4>();
	resDesc.res.pitch2D.devPtr = d_image;
	resDesc.res.pitch2D.height = sizeh;
	resDesc.res.pitch2D.width = sizew;
	resDesc.res.pitch2D.pitchInBytes = pitch;

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
	cudaMalloc((void **)&d_new_image, neww * newh * sizeof(float4));

	dim3 blocksize(16, 16);
	dim3 gridsize(neww / blocksize.x, newh / blocksize.y);

	float width_norm = (1.0f / float(neww-1));
	float height_norm = (1.0f / float(newh-1));
	resize_kernel<<<gridsize, blocksize>>>(tex, d_new_image, width_norm, height_norm);

	// copy c back to host
	cudaMemcpy(new_image, d_new_image, neww * newh * sizeof(float4), cudaMemcpyDeviceToHost);

	// clean up
	cudaFree(d_new_image);

	cudaDestroyTextureObject(tex);
	//cudaFreeArray(cuArray);
	cudaFree(d_image);
}

}
