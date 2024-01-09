/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*****************
*  rtk #includes *
*****************/
//#include "rtkCudaUtilities.hcu"
//#include "rtkConfiguration.h"
//#include "rtkCudaFDKBackProjectionImageFilter.hcu"


/*****************
*  C   #includes *
*****************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*****************
* CUDA #includes *
*****************/
#include <cuda.h>
#include "../common/cudaLib.cuh"
#include "backprojection.cuh"
#include "cudaBpLib.cuh"
#include "../common/type.h"



// T E X T U R E S ////////////////////////////////////////////////////////
#if USE_8BITS == 1
texture<float, cudaTextureType2DLayered> tex_proj(false, cudaFilterModeLinear, cudaAddressModeClamp);;
#else
texture<float, cudaTextureType2DLayered> tex_proj(false, cudaFilterModePoint, cudaAddressModeClamp);;
#endif

texture<float, 3, cudaReadModeElementType> tex_proj_3D;

static const int SLAB_SIZE = cudaBackProjection::MAX_PROJ;

// Constant memory
__constant__ float c_matrices[SLAB_SIZE * 12]; //Can process stacks of at most SLAB_SIZE projections
__constant__ int3 c_projSize;
__constant__ int3 c_vol_size;

inline __host__ __device__ float3 matrix_multiply(float3 a, float* matrix)
{
	return make_float3(matrix[0] * a.x + matrix[1] * a.y + matrix[2] * a.z + matrix[3],
		matrix[4] * a.x + matrix[5] * a.y + matrix[6] * a.z + matrix[7],
		matrix[8] * a.x + matrix[9] * a.y + matrix[10] * a.z + matrix[11]);
}

inline std::pair<int, int> GetCudaComputeCapability(int device)
{
	struct cudaDeviceProp properties;
	if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
		std::cout << "Invalid CUDA device";
	return std::make_pair(properties.major, properties.minor);
}
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
template<typename Texture>
__device__ __forceinline__ float RtkSUBPIXEL(Texture tex, int width, int height, float u, float v, int layer)
{
	float s;
#if 1 == USE_8BITS
	s = tex2DLayered(tex, u + 0.5, v + 0.5, layer);
#else
	if (u >= 0 && u < width - 1 && v >= 0 && v < height - 1) {
		float iu = floorf(u);
		float iv = floorf(v);
		float du = u - iu;
		float dv = v - iv;
		float _du = 1.0 - du;
		float _dv = 1.0 - dv;
		float x0 = tex2DLayered(tex, iu, iv, layer);
		float x1 = tex2DLayered(tex, iu + 1, iv, layer);
		float x2 = tex2DLayered(tex, iu, iv + 1, layer);
		float x3 = tex2DLayered(tex, iu + 1, iv + 1, layer);
		x0 = FMAD(x0, _du, x1*du);
		x2 = FMAD(x2, _du, x3*du);
		s = FMAD(x0, _dv, x2*dv);
		//x0 = x0*_du + x1*du;
		//x2 = x2*_du + x3*du;
		//s = x0*_dv + x2*dv;
	} else {
		s = 0;
	}
#endif
	return s;
}

__global__
void kernel_fdk(float *dev_vol_in, float *dev_vol_out, unsigned int Blocks_Y)
{
  // CUDA 2.0 does not allow for a 3D grid, which severely
  // limits the manipulation of large 3D arrays of data.  The
  // following code is a hack to bypass this implementation
  // limitation.
  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= c_vol_size.x || j >= c_vol_size.y || k >= c_vol_size.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*c_vol_size.y)*(c_vol_size.x);

  float3 ip;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // matrix multiply
    ip = matrix_multiply(make_float3(i,j,k), &(c_matrices[12*proj]));

    // Change coordinate systems
    ip.z = 1 / ip.z;
    ip.x = ip.x * ip.z;
    ip.y = ip.y * ip.z;

    // Get texture point, clip left to GPU
    voxel_data += tex3D(tex_proj_3D, ip.x, ip.y, proj + 0.5) *  ip.z * ip.z;
    }

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

__global__
void kernel_fdk_3Dgrid(float *dev_vol_in, float * dev_vol_out)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_vol_size.x || j >= c_vol_size.y || k >= c_vol_size.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*c_vol_size.y)*(c_vol_size.x);

  float3 ip;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // matrix multiply
    ip = matrix_multiply(make_float3(i,j,k), &(c_matrices[12*proj]));

    // Change coordinate systems
    ip.z = 1 / ip.z;
    ip.x = ip.x * ip.z;
    ip.y = ip.y * ip.z;

    // Get texture point, clip left to GPU, and accumulate in voxel_data
    //voxel_data += tex2DLayered(tex_proj, ip.x, ip.y, proj) *  ip.z * ip.z;
	voxel_data += RtkSUBPIXEL(tex_proj, c_projSize.x, c_projSize.y, ip.x, ip.y, proj) *  ip.z * ip.z;
    }

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_back_project /////////////////////////////
void BackProjectionRTK::CUDA_reconstruct_conebeam(
  int proj_size[3],
  int vol_size[3],
  float *matrices,
  float *dev_vol_in,
  float *dev_vol_out,
  float *dev_proj)
{
  int device;
  cudaGetDevice(&device);

  // Copy the size of inputs into constant memory
  cudaMemcpyToSymbol(c_projSize, proj_size, sizeof(int3));
  cudaMemcpyToSymbol(c_vol_size, vol_size, sizeof(int3));

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_matrices, &(matrices[0]), 12 * sizeof(float) * proj_size[2]);

#if 0
  // set texture parameters
  tex_proj.addressMode[0] = cudaAddressModeBorder;
  tex_proj.addressMode[1] = cudaAddressModeBorder;
  tex_proj.addressMode[2] = cudaAddressModeBorder;
  //tex_proj.filterMode = cudaFilterModeLinear; 
  tex_proj.filterMode = cudaFilterModePoint;
  tex_proj.normalized = false; // don't access with normalized texture coords

  tex_proj_3D.addressMode[0] = cudaAddressModeBorder;
  tex_proj_3D.addressMode[1] = cudaAddressModeBorder;
  tex_proj_3D.addressMode[2] = cudaAddressModeBorder;
  tex_proj_3D.filterMode = cudaFilterModeLinear;
  tex_proj_3D.normalized = false; // don't access with normalized texture coords
#endif

  // Copy projection data to array, bind the array to the texture
  cudaExtent projExtent = make_cudaExtent(proj_size[0], proj_size[1], proj_size[2]);
  //cudaArray *array_proj;
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK_ERROR;

  // Allocate array for input projections, in order to bind them to
  // either a 2D layered texture (requires GetCudaComputeCapability >= 2.0) or
  // a 3D texture
  if (NULL == array_proj) {
	  if (CUDA_VERSION < 4000 || GetCudaComputeCapability(device).first <= 1)
		  cudaMalloc3DArray((cudaArray**)&array_proj, &channelDesc, projExtent);
	  else
		  cudaMalloc3DArray((cudaArray**)&array_proj, &channelDesc, projExtent, cudaArrayLayered);
	  CUDA_CHECK_ERROR;
  }

  // Copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr(dev_proj, proj_size[0]*sizeof(float), proj_size[0], proj_size[1]);
  copyParams.dstArray = (cudaArray*)array_proj;
  copyParams.extent   = projExtent;
  copyParams.kind     = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&copyParams);
  CUDA_CHECK_ERROR;

  // Thread Block Dimensions
  const int tBlock_x = 32;
  const int tBlock_y = 4;
  const int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int  blocksInX = (vol_size[0]-1)/tBlock_x + 1;
  unsigned int  blocksInY = (vol_size[1]-1)/tBlock_y + 1;
  unsigned int  blocksInZ = (vol_size[2]-1)/tBlock_z + 1;

  // Run kernels. Note: Projection data is passed via texture memory,
  // transform matrix is passed via constant memory
  if(CUDA_VERSION<4000 || GetCudaComputeCapability(device).first<=1)
    {
	  VERIFY_TRUE(0);
    // Compute block and grid sizes
    dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    // Bind the array of projections to a 3D texture
    cudaBindTextureToArray(tex_proj_3D, (cudaArray*)array_proj, channelDesc);
    CUDA_CHECK_ERROR;

    kernel_fdk <<< dimGrid, dimBlock >>> ( dev_vol_in,
                                           dev_vol_out,
                                           blocksInY );
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR;
    // Unbind the image and projection matrix textures
    cudaUnbindTexture (tex_proj_3D);
    CUDA_CHECK_ERROR;
    }
  else
    {
    // Compute block and grid sizes
    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
    CUDA_CHECK_ERROR;

	// Bind the array of projections to a 2D layered texture
	cudaBindTextureToArray(tex_proj, (cudaArray*)array_proj, channelDesc);
	CUDA_CHECK_ERROR;

	{
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		kernel_fdk_3Dgrid << < dimGrid, dimBlock >> > (dev_vol_in,
			dev_vol_out);
		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR;

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);

		fKernelTime += time;
	}
	// Unbind the image and projection matrix textures
	cudaUnbindTexture(tex_proj);
	CUDA_CHECK_ERROR;
  }


  //// Cleanup
  //cudaFreeArray ((cudaArray*)array_proj);
  //CUDA_CHECK_ERROR;
}

////////////////////////////////////

BackProjectionRTK::BackProjectionRTK(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, bool _bDualBuffer /*= false*/) {
	DISPLAY_FUNCTION;
	printf("Precision : %s \n", USE_8BITS == 1 ? "8bits" : "32bit");
	std::cout << "MAX_PROJ = " << MAX_PROJ << std::endl;
	int tm = timeGetTime();
	bDualBuffer = _bDualBuffer;
	projWidth = _projWidth; projHeight = _projHeight; projCount = _projCount;
	nx = _nx; ny = _ny; nz = _nz;
	//pDevProjData = std::make_shared<GpuArray3D<float>>(_projWidth, _projHeight, _projCount, cudaArrayLayered);
	pDevVolIn = std::make_shared<GpuBuffer<float>>(nx, ny, nz);
	pDevVolOut = std::make_shared<GpuBuffer<float>>(nx, ny, nz);
	pDevVolIn->Zero();
	pDevVolOut->Zero();
	array_proj = NULL;
	{
		int device;
		cudaGetDevice(&device);
		// Copy projection data to array, bind the array to the texture
		cudaExtent projExtent = make_cudaExtent(projWidth, projHeight, projCount);
		//cudaArray *array_proj;
		static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		CUDA_CHECK_ERROR;

		if (NULL == array_proj) {
			if (CUDA_VERSION < 4000 || GetCudaComputeCapability(device).first <= 1)
				cudaMalloc3DArray((cudaArray**)&array_proj, &channelDesc, projExtent);
			else
				cudaMalloc3DArray((cudaArray**)&array_proj, &channelDesc, projExtent, cudaArrayLayered);
			CUDA_CHECK_ERROR;
		}
	}
	tm = timeGetTime() - tm;
	std::cout << "init time = " << tm << " ms" << std::endl;
}

BackProjectionRTK::~BackProjectionRTK() {
	if (array_proj) {
		cudaFreeArray((cudaArray*)array_proj);
		CUDA_CHECK_ERROR;
	}
}
bool BackProjectionRTK::BP(int TRANSPOSE, float* pProjData, int width, int height, float* pProjMat, int nProj) {
	//VERIFY_TRUE(cudaSuccess == cudaFuncSetCacheConfig(cudaBP<TRANSPOSE>, cudaFuncCachePreferL1));
	VERIFY_TRUE(TRANSPOSE == 0);
	this->SwapInOutBuffer();
	int proj_size[3] = { width, height, nProj };
	int vol_size[3] = {nx, ny, nz};
	float *matrices = pProjMat;
	float* dev_vol_in = pDevVolIn->GetData();
	float* dev_vol_out = pDevVolOut->GetData();
	float* dev_proj = pProjData;
	CUDA_reconstruct_conebeam(proj_size, vol_size, matrices, dev_vol_in, dev_vol_out, dev_proj);

	return true;
}

bool BackProjectionRTK::GetVolumeData(float* pVol, int size) {
	pDevVolOut->CopyToHost(pVol, nx, nx, ny, nz);
	CUDA_CHECK_ERROR;
	return true;
}

inline void BackProjectionRTK::SwapInOutBuffer() {
	auto tmp = this->pDevVolIn;
	pDevVolIn = pDevVolOut;
	pDevVolOut = tmp;
}
