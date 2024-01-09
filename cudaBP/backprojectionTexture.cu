#include "../common/type.h"
#include "backprojection.cuh"
#include "../common/cudaLib.cuh"
#include "cudaBpLib.cuh"

const int NP = BackProjectionTexture::MAX_PROJ; //batch count of processing proj-data   
__constant__ float4 ProjMat[NP][3]; //NPx3x4 proj-matrix
//textures for proj-data sequence
//texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModeLinear, cudaAddressModeClamp);
//texture<float, cudaTextureType3D> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
//texture tex_ij; //1-D texture for (i,j) index array



__device__ __forceinline__ float SUBPIXEL(float u, float v, int layer)
{
	float s;
#if 1
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



template<int TRANSPOSE, bool bDualBuffer>
__global__ void cudaBP(const float* volIn, float* volOut, int nK, const float2* pIJ, int nIJ, int2 img_dim, int nProj)
{
	__shared__ float2 IJ; //share (i,j) in a block
	if (threadIdx.x == 0) //load (i,j) only once 
		IJ = __ldg(&pIJ[blockIdx.x]); //access via cache
	__syncthreads();

	//U, F, F2 are shared in a block
	__shared__ float U[NP], F[NP], F2[NP];
	int  k = threadIdx.x;
	int _k = nK - 1 - k;  //geometry symmetric
	float4 IJKW = make_float4(IJ.x, IJ.y, k, 1.f);
	//in parallel compute U, F, F2
	if (k < nProj) {
		//use register
		float z = 1.f / dotIJKW(ProjMat[k][2], IJKW); //inner product
		F[k] = z;
		U[k] = dotIJKW(ProjMat[k][0], IJKW)*z;
		F2[k] = z * z;
	}
	__syncthreads();
	assert(k < nK/2);
	//in-register accumulator,
	float sum = 0, _sum = 0;
	//geometry symmetric
	float v, _v;

	#pragma unroll (16)
	for (int i = 0; i<nProj; i++) {
		v = dotIJKW(ProjMat[i][1], IJKW)*F[i];
		if (TRANSPOSE == 0){
			_v   = img_dim.y - 1 - v;       //geometry symmetric		
			sum  = FMAD(SUBPIXEL(U[i], v, i), F2[i], sum);
			_sum = FMAD(SUBPIXEL(U[i], _v, i), F2[i], _sum);
		}
		if (TRANSPOSE == 1){
			_v   = img_dim.x - 1 - v;      //geometry symmetric
			sum  = FMAD(SUBPIXEL(v, U[i], i), F2[i], sum);
			_sum = FMAD(SUBPIXEL(_v, U[i], i), F2[i], _sum);			
		} 
	}
	//update volume
	uint offset = nK * blockIdx.x;
	float* pVolOut = volOut + offset;
	if (bDualBuffer == 1) {
		const float* pVolIn = volIn + offset;
		pVolOut[k] = pVolIn[k] + sum;
		pVolOut[_k] = pVolIn[_k] + _sum;
	}else {
		pVolOut[k] += sum;
		pVolOut[_k] += _sum;
	}
}


///////////////////////////////////////////////////
BackProjectionTexture::BackProjectionTexture(int _nK, int _projWidth, int _projHeight, int _projCount, const int* pIJ, int _nIJ, bool _bDualBuffer/* = true*/)
{
	DISPLAY_FUNCTION;
	std::cout << "MAX_PROJ = " << MAX_PROJ << std::endl;
	int tm = timeGetTime();
	nK = _nK;
	projWidth = _projWidth; _projHeight = _projHeight;  _projCount = _projCount; nIJ = _nIJ;
	bDualBuffer = _bDualBuffer;
	vecIJ.resize(nIJ*2);
	vecfIJ.resize(nIJ * 2);
	for (int i = 0; i < nIJ * 2; i++) {
		vecfIJ[i] = vecIJ[i] = pIJ[i];		
	}

	pDevIJ = std::make_shared<GpuData<float2>>(nIJ);
	pDevIJ->CopyFromHost((float2*)&vecfIJ[0], nIJ, nIJ, 1);
	pDevProjData = std::make_shared<GpuArray3D<float>>(_projWidth, _projHeight, _projCount, cudaArrayLayered);
	//pDevProjData->Zero();
	pDevVolOut = std::make_shared<GpuData<float>>(nK, nIJ);
	pDevVolOut->Zero();
	if (bDualBuffer) {
		pDevVolIn = std::make_shared<GpuData<float>>(nK, nIJ);
		pDevVolIn->Zero();
	}else {
		pDevVolIn = pDevVolOut;
	}
	CUDA_CHECK_ERROR;	
	tm = timeGetTime() - tm;
	printf("DualBuffer = %s\n", bDualBuffer ? "true" : "false");
	std::cout << "init time = " << tm << " ms" << std::endl;
}

BackProjectionTexture::~BackProjectionTexture() {

}

bool BackProjectionTexture::BP(int TRANSPOSE, const float* pProjData, int width, int height, float* pProjMat, int nProj) {
	if (TRANSPOSE == 0) return inBP<0>(pProjData, width, height, pProjMat, nProj);
	if (TRANSPOSE == 1) return inBP<1>(pProjData, width, height, pProjMat, nProj);
}

template<int TRANSPOSE>
bool BackProjectionTexture::inBP(const float* pProjData, int width, int height, float* pProjMat, int nProj) {
	//VERIFY_TRUE(cudaSuccess == cudaFuncSetCacheConfig(cudaBP<TRANSPOSE>, cudaFuncCachePreferL1));
	if (bDualBuffer) SwapInOutBuffer();
	LoadConstant(&ProjMat[0][0], pProjMat, PROJ_MAT_SIZE*nProj);
	pDevProjData->CopyFromHost(pProjData, width, width, height, nProj);
	pDevProjData->BindToTexture(&tex);
	dim3 blockSize(Align(nK / 2, 32), 1, 1);
	dim3 gridSize(nIJ, 1, 1);
	int2 dimImg = { width, height };
	{
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		//cudaBP<TRANSPOSE> << <gridSize, blockSize >> > (pDevVol->GetData(), nK, pDevIJ->GetData(), nIJ, dimImg, nProj);
		if (bDualBuffer)
			cudaBP<TRANSPOSE, true> << <gridSize, blockSize >> > (pDevVolIn->GetData(), pDevVolOut->GetData(), nK, pDevIJ->GetData(), nIJ, dimImg, nProj);
		else
			cudaBP<TRANSPOSE, false> << <gridSize, blockSize >> > (pDevVolIn->GetData(), pDevVolOut->GetData(), nK, pDevIJ->GetData(), nIJ, dimImg, nProj);

		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR;
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);

		fKernelTime += time;
	}
	return true;
}

bool BackProjectionTexture::GetVolumeData(float* pVol, int size) {
	VERIFY_TRUE(size >= nK * nIJ);
	pDevVolOut->CopyToHost(pVol, nK, nK, nIJ);
	return true;
}

inline void BackProjectionTexture::SwapInOutBuffer() {
	VERIFY_TRUE(pDevVolIn);
	auto tmp = this->pDevVolIn;
	pDevVolIn = pDevVolOut;
	pDevVolOut = tmp;
}


//////////////////////////////////////////
template
bool BackProjectionTexture::inBP<0>(const float* pProjData, int width, int height, float* pProjMat, int count);
template
bool BackProjectionTexture::inBP<1>(const float* pProjData, int width, int height, float* pProjMat, int count);
