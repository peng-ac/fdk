#include "../common/type.h"
#include "backprojection.cuh"
#include "../common/cudaLib.cuh"
#include "cudaBpLib.cuh"


const int NP = BackProjectionGlobalmem::MAX_PROJ; //batch count of processing proj-data   
__constant__ float4 ProjMat[NP][3]; //NPx3x4 proj-matrix
//textures for proj-data sequence
//texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
//texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModeLinear, cudaAddressModeClamp);
//texture<float, cudaTextureType3D> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
//texture tex_ij; //1-D texture for (i,j) index array

__device__ __forceinline__ float GMemSUBPIXEL(const float* pProj, int nProjPitch, int width, int height, float u, float v)
{
	float s;
	if (u >= 0 && u < width - 1 && v >= 0 && v < height - 1) {
		float iu = floorf(u);
		float iv = floorf(v);
		int   nu = iu;
		int   nv = iv;
		float du = u - iu;
		float dv = v - iv;
		float _du = 1.0 - du;
		float _dv = 1.0 - dv;

		const float* p0 = pProj + nv * nProjPitch + nu;
		const float* p1 = p0 + nProjPitch;

		float x0 = __ldg(&p0[0]);
		float x1 = __ldg(&p0[1]);
		float x2 = __ldg(&p1[0]);
		float x3 = __ldg(&p1[1]);

		x0 = FMAD(x0, _du, x1*du);
		x2 = FMAD(x2, _du, x3*du);
		s = FMAD(x0, _dv, x2*dv);
		//x0 = x0*_du + x1*du;
		//x2 = x2*_du + x3*du;
		//s = x0*_dv + x2*dv;
	}else {
		s = 0;
	}

	return s;
}

template<int TRANSPOSE, bool bDualBuffer>
__global__ void cudaBP(const float* volIn, float* volOut, int nK, const float2* pIJ, int nIJ, const float* pProj, int nProjPitch, int3 img_dim)
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
	const int nProj = img_dim.z;
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

	const int nFrameBufferSize = nProjPitch * img_dim.y;
	const float* _pProj = pProj;
	//#pragma unroll (16)
	for (int i = 0; i<nProj; i++) {
		v = dotIJKW(ProjMat[i][1], IJKW)*F[i];
		if (TRANSPOSE == 0){
			_v   = img_dim.y - 1 - v;       //geometry symmetric		
			sum  = FMAD(GMemSUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, U[i], v), F2[i], sum);
			_sum = FMAD(GMemSUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, U[i], _v), F2[i], _sum);
		}
		if (TRANSPOSE == 1){
			_v   = img_dim.x - 1 - v;      //geometry symmetric
			sum  = FMAD(GMemSUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y,   v, U[i]), F2[i], sum);
			_sum = FMAD(GMemSUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y,  _v, U[i]), F2[i], _sum);
		} 
		_pProj += nFrameBufferSize;
	}
	//update volume
	uint offset = nK * blockIdx.x;
	float* pVolOut = volOut + offset;
	if (bDualBuffer == 1) {
		const float* pVolIn = volIn + offset;
		pVolOut[ k] = __ldg(&pVolIn[ k]) + sum;
		pVolOut[_k] = __ldg(&pVolIn[_k]) + _sum;
	} else {
		pVolOut[k] += sum;
		pVolOut[_k] += _sum;
	}
}


///////////////////////////////////////////////////
BackProjectionGlobalmem::BackProjectionGlobalmem(int _nK, int _projWidth, int _projHeight, int _projCount, const int* pIJ, int _nIJ, bool _bDualBuffer /*= false*/)
{
	DISPLAY_FUNCTION;
	printf("Precision : %s \n", "32bit");
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
	pDevProjData = std::make_shared<GpuData<float>>(_projWidth, _projHeight, _projCount);
	//pDevProjData->Zero();
	pDevVolOut = std::make_shared<GpuData<float>>(nK, nIJ);
	pDevVolOut->Zero();
	if (bDualBuffer) {
		pDevVolIn = std::make_shared<GpuData<float>>(nK, nIJ);
		pDevVolIn->Zero();
	}else{
		pDevVolIn = pDevVolOut;
	}
	CUDA_CHECK_ERROR;
	tm = timeGetTime() - tm;
	printf("DualBuffer = %s\n", bDualBuffer?"true":"false");
	std::cout << "init time = " << tm << " ms" << std::endl;
}

BackProjectionGlobalmem::~BackProjectionGlobalmem() {

}

bool BackProjectionGlobalmem::BP(int TRANSPOSE, const float* pProjData, int width, int height, float* pProjMat, int nProj) {
	if (TRANSPOSE == 0) return inBP<0>(pProjData, width, height, pProjMat, nProj);
	if (TRANSPOSE == 1) return inBP<1>(pProjData, width, height, pProjMat, nProj);
}

template<int TRANSPOSE>
bool BackProjectionGlobalmem::inBP(const float* pProjData, int width, int height, float* pProjMat, int nProj) {
	//VERIFY_TRUE(cudaSuccess == cudaFuncSetCacheConfig(cudaBP<TRANSPOSE>, cudaFuncCachePreferL1));
	if (bDualBuffer) SwapInOutBuffer();
	LoadConstant(&ProjMat[0][0], pProjMat, PROJ_MAT_SIZE*nProj);
	pDevProjData->CopyFromHost(pProjData, width, width, height, nProj);
	//pDevProjData->BindToTexture(&tex);
	dim3 blockSize(Align(nK / 2, WarpSize), 1, 1);
	dim3 gridSize(nIJ, 1, 1);
	int3 dimImg = { width, height, nProj };
	{
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		if (bDualBuffer)
			cudaBP<TRANSPOSE, true> << <gridSize, blockSize >> > (pDevVolIn->GetData(), pDevVolOut->GetData(), nK, pDevIJ->GetData(), nIJ, pDevProjData->GetData(), pDevProjData->DataPitch(), dimImg);
		else
			cudaBP<TRANSPOSE, false> << <gridSize, blockSize >> > (pDevVolIn->GetData(), pDevVolOut->GetData(), nK, pDevIJ->GetData(), nIJ, pDevProjData->GetData(), pDevProjData->DataPitch(), dimImg);

		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR;
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);

		fKernelTime += time;
	}
	return true;
}

bool BackProjectionGlobalmem::GetVolumeData(float* pVol, int size) {
	VERIFY_TRUE(size >= nK * nIJ);
	pDevVolOut->CopyToHost(pVol, nK, nK, nIJ);
	return true;
}

inline void BackProjectionGlobalmem::SwapInOutBuffer() {
	VERIFY_TRUE(pDevVolIn);
	auto tmp = this->pDevVolIn;
	pDevVolIn = pDevVolOut;
	pDevVolOut = tmp;
}


//////////////////////////////////////////
template
bool BackProjectionGlobalmem::inBP<0>(const float* pProjData, int width, int height, float* pProjMat, int count);
template
bool BackProjectionGlobalmem::inBP<1>(const float* pProjData, int width, int height, float* pProjMat, int count);
