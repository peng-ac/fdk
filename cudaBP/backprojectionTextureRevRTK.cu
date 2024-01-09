#include "../common/type.h"
#include "backprojection.cuh"
#include "../common/cudaLib.cuh"
#include "cudaBpLib.cuh"


const int NP = BackProjectionTextureRevRTK::MAX_PROJ; //batch count of processing proj-data   
__constant__ float4 ProjMat[NP][3]; //NPx3x4 proj-matrix
									//textures for proj-data sequence
									//texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
									//texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModeLinear, cudaAddressModeClamp);
									//texture<float, cudaTextureType3D> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
									//texture tex_ij; //1-D texture for (i,j) index array
#if USE_8BITS == 1
texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModeLinear, cudaAddressModeClamp);
#else
texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
#endif

__device__ __forceinline__ float TexRevRtkSUBPIXEL(const float* pProj, int nProjPitch, int width, int height, float u, float v, int layer)
{
	assert(pProj == 0);
	float s;
#if 1 == USE_8BITS
	s = tex2DLayered(tex, u + 0.5, v + 0.5, layer);
#else
	float iu = floorf(u);
	float iv = floorf(v);
	float du = u - iu;
	float dv = v - iv;
	float _du = 1.0 - du;
	float _dv = 1.0 - dv;
	float x0 = tex2DLayered(tex, iu, iv, layer);
	float x1 = tex2DLayered(tex, iu + 1.f, iv, layer);
	float x2 = tex2DLayered(tex, iu, iv + 1.f, layer);
	float x3 = tex2DLayered(tex, iu + 1.f, iv + 1.f, layer);
	x0 = FMAD(x0, _du, x1*du);
	x2 = FMAD(x2, _du, x3*du);
	s = FMAD(x0, _dv, x2*dv);
	//x0 = x0*_du + x1*du;
	//x2 = x2*_du + x3*du;
	//s = x0*_dv + x2*dv;
#endif
	return s;
}

template<int TRANSPOSE, bool bDualBuffer, bool bSymmetric>
__global__ void cudaBP(const float* volIn, float* volOut, int3 vol_dim, int3 img_dim)
{
	const uint laneId = threadIdx.x;
	uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

	if (i >= vol_dim.z || j >= vol_dim.y || k >= vol_dim.x)
		return;

	//int _k = nK - 1 - k;  //geometry symmetric
	float4 IJKW = make_float4(k, j, i, 1.f);
	const int nProj = img_dim.z;
	float Z, U, F2;

	assert(nProj <= 32);
	//in parallel compute U, F, F2
	if (laneId < nProj) {
		//use register
		Z = 1.f / dotIJKW(ProjMat[laneId][2], IJKW); //inner product
		U = dotIJKW(ProjMat[laneId][0], IJKW)*Z;
		F2 = Z * Z;
	}

	//in-register accumulator,
	float sum  = 0, _sum = 0;
	//geometry symmetric
	float v, _v;
	int _i;
	if (bSymmetric) _i = vol_dim.z - 1 - i;

	//const int nFrameBufferSize = nProjPitch * img_dim.y;
	const float* _pProj = NULL;
	const int nProjPitch = 0;
	#pragma unroll (8)
	for (int s = 0; s<nProj; s++) {
		float u = __my_shfl(U, s);
		float f = __my_shfl(Z, s);
		float f2 = __my_shfl(F2, s);
		//float f2 = f * f;
		v = dotIJKW(ProjMat[s][1], IJKW)*f;
		if (TRANSPOSE == 0) {
			if (bSymmetric)
				_v = img_dim.y - 1 - v;       //geometry symmetric	
			sum = FMAD(TexRevRtkSUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, u, v, s), f2, sum);
			if (bSymmetric)
				_sum = FMAD(TexRevRtkSUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, u, _v, s), f2, _sum);
		}
		if (TRANSPOSE == 1) {
			if (bSymmetric) 
				_v = img_dim.x - 1 - v;      //geometry symmetric
			sum = FMAD(TexRevRtkSUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, v, u, s), f2, sum);
			if (bSymmetric) 
				_sum = FMAD(TexRevRtkSUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, _v, u, s), f2, _sum);
		}
		//_pProj += nFrameBufferSize;
	}
	//update volume
	//uint offset = k*vol_dim.z*vol_dim.y + j*vol_dim.z;
	uint offset = (k * vol_dim.y + j) * vol_dim.z;
	float* pVolOut = volOut + offset;
	if (bDualBuffer == 1) {
		const float* pVolIn = volIn + offset;
		pVolOut[i] = __ldg(&pVolIn[i]) + sum;
		if (bSymmetric) pVolOut[_i] = __ldg(&pVolIn[_i]) + _sum;
	} else {
		pVolOut[i] += sum;
		if (bSymmetric) pVolOut[_i] += _sum;
		//pVolOut[_k] += _sum;
	}
}


///////////////////////////////////////////////////
BackProjectionTextureRevRTK::BackProjectionTextureRevRTK(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, bool _bDualBuffer/* = false*/):cudaBackProjection()
{
	DISPLAY_FUNCTION;
	printf("Precision : %s \n", USE_8BITS == 1 ? "8bits" : "32bit");
	std::cout << "MAX_PROJ = " << MAX_PROJ << std::endl;
	int tm = timeGetTime();
	nx = _nx; ny = _ny; nz = _nz;
	projWidth = _projWidth; _projHeight = _projHeight;  _projCount = _projCount; 
	bDualBuffer = _bDualBuffer;

	pDevProjData = std::make_shared<GpuArray3D<float>>(_projWidth, _projHeight, _projCount, cudaArrayLayered);
	//pDevProjData->Zero();
	pDevVolOut = std::make_shared<GpuBuffer<float>>(nz, ny, nx);
	pDevVolOut->Zero();
	if (bDualBuffer) {
		pDevVolIn = std::make_shared<GpuBuffer<float>>(nz, ny, nx);
		pDevVolIn->Zero();
	}else {
		pDevVolIn = pDevVolOut;
	}
	CUDA_CHECK_ERROR;
	tm = timeGetTime() - tm;
	printf("DualBuffer = %s\n", bDualBuffer ? "true" : "false");
	std::cout << "init time = " << tm << " ms" << std::endl;
}

BackProjectionTextureRevRTK::~BackProjectionTextureRevRTK() {

}

bool BackProjectionTextureRevRTK::BP(int TRANSPOSE, const float* pProjData, int width, int height, float* pProjMat, int nProj) {
	const bool bSymmetric = true;
	if (TRANSPOSE == 0) return inBP<0, bSymmetric>(pProjData, width, height, pProjMat, nProj);
	if (TRANSPOSE == 1) return inBP<1, bSymmetric>(pProjData, width, height, pProjMat, nProj);
}

template<int TRANSPOSE, bool bSymmetric>
bool BackProjectionTextureRevRTK::inBP(const float* pProjData, int width, int height, float* pProjMat, int nProj) {
	//VERIFY_TRUE(cudaSuccess == cudaFuncSetCacheConfig(cudaBP<TRANSPOSE>, cudaFuncCachePreferL1));
	if (bDualBuffer) SwapInOutBuffer();
	LoadConstant(&ProjMat[0][0], pProjMat, PROJ_MAT_SIZE*nProj);
	pDevProjData->CopyFromHost(pProjData, width, width, height, nProj);
	pDevProjData->BindToTexture(&tex);
	dim3 blockSize(WarpSize, 4, 4);
	dim3 gridSize(UpDivide(nz/2, blockSize.x), UpDivide(ny, blockSize.y), UpDivide(nx, blockSize.z));
	if (!bSymmetric) gridSize.x = UpDivide(nz, blockSize.x);
	int3 vol_dim = {nx, ny, nz};
	int3 dimImg = { width, height, nProj };
	{
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		if (bDualBuffer)
			cudaBP<TRANSPOSE, true, bSymmetric> << <gridSize, blockSize >> > (pDevVolIn->GetData(), pDevVolOut->GetData(), vol_dim, dimImg);
		else
			cudaBP<TRANSPOSE, false, bSymmetric> << <gridSize, blockSize >> > (pDevVolIn->GetData(), pDevVolOut->GetData(), vol_dim, dimImg);

		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR;

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);

		fKernelTime += time;
	}
	return true;
}

bool BackProjectionTextureRevRTK::GetVolumeData(float* pVol, int size) {
	VERIFY_TRUE(size >= nx*ny*nz);
	pDevVolOut->CopyToHost(pVol, nz, nz, ny, nx);
	return true;
}

inline void BackProjectionTextureRevRTK::SwapInOutBuffer() {
	VERIFY_TRUE(pDevVolIn);
	auto tmp = this->pDevVolIn;
	pDevVolIn = pDevVolOut;
	pDevVolOut = tmp;
}


////////////////////////////////////////////
//template
//bool BackProjectionTextureRevRTK::inBP<0>(const float* pProjData, int width, int height, float* pProjMat, int count);
//template
//bool BackProjectionTextureRevRTK::inBP<1>(const float* pProjData, int width, int height, float* pProjMat, int count);
