#include "../common/type.h"
#include "backprojection.cuh"
#include "cudaLib.cuh"

const int NP = BackProjection::MAX_PROJ; //batch count of processing proj-data   
__constant__ float4 ProjMat[NP][3]; //NPx3x4 proj-matrix
//textures for proj-data sequence
texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
//texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModeLinear, cudaAddressModeClamp);
//texture<float, cudaTextureType3D> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
//texture tex_ij; //1-D texture for (i,j) index array

__device__ __forceinline__ float FMAD(const float a, const float b, const float c) {
#if 1
	float d;
	asm volatile("mad.rz.ftz.f32 %0,%1,%2,%3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
	return d;	
#else
	return a * b + c;
#endif
}

__device__ __forceinline__ float SUBPIXEL(float u, float v, int layer)
{
	float s;
#if 0
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

__device__ __forceinline__ float dot(const float4 a, const float4 b) {
	float sum = 0;
#if 0
	sum = a.x*b.x + sum;
	sum = a.y*b.y + sum;
	sum = a.z*b.z + sum;
	sum = a.w*b.w + sum;
#else
	sum = FMAD(a.x, b.x, sum);
	sum = FMAD(a.y, b.y, sum);
	sum = FMAD(a.z, b.z, sum);
	sum = FMAD(a.w, b.w, sum);
#endif
	return sum;
}

template<int TRANSPOSE>
__global__ void cudaBP(float* vol, int nK, const float2* pIJ, int nIJ, int2 img_dim, int nProj)
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
		float z = 1.f / dot(ProjMat[k][2], IJKW); //inner product
		F[k] = z;
		U[k] = dot(ProjMat[k][0], IJKW)*z;
		F2[k] = z * z;
	}
	__syncthreads();
	assert(k < nK/2);
	//in-register accumulator,
	float sum = 0, _sum = 0;
	//geometry symmetric
	float v, _v;

	#pragma unroll (8)
	for (int i = 0; i<nProj; i++) {
		v = dot(ProjMat[i][1], IJKW)*F[i];
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
	float* pVol = vol + nK * blockIdx.x;
	pVol[ k] += sum;
	pVol[_k] += _sum;
}


///////////////////////////////////////////////////
BackProjection::BackProjection(int _nK, int _projWidth, int _projHeight, int _projCount, const int* pIJ, int _nIJ)
{
	std::cout << "MAX_PROJ = " << MAX_PROJ << std::endl;
	nK = _nK;
	projWidth = _projWidth; _projHeight = _projHeight;  _projCount = _projCount; nIJ = _nIJ;
	vecIJ.resize(nIJ*2);
	vecfIJ.resize(nIJ * 2);
	for (int i = 0; i < nIJ * 2; i++) {
		vecfIJ[i] = vecIJ[i] = pIJ[i];		
	}

	pDevIJ = std::make_shared<DevData<float2>>(nIJ);
	pDevIJ->CopyFromHost((float2*)&vecfIJ[0], nIJ, nIJ, 1);
	pDevProjData = std::make_shared<DevArray3D<float>>(_projWidth, _projHeight, _projCount, cudaArrayLayered);
	//pDevProjData->Zero();
	pDevVol = std::make_shared<DevData<float>>(nK, nIJ);
	pDevVol->Zero();
	CUDA_CHECK_ERROR;
}

BackProjection::~BackProjection() {

}

template<int TRANSPOSE>
bool BackProjection::BP(const float* pProjData, int width, int height, float* pProjMat, int nProj) {
	//VERIFY_TRUE(cudaSuccess == cudaFuncSetCacheConfig(cudaBP<TRANSPOSE>, cudaFuncCachePreferL1));
	LoadConstant(&ProjMat[0][0], pProjMat, PROJ_MAT_SIZE*nProj);
	pDevProjData->CopyFromHost(pProjData, width, width, height, nProj);
	pDevProjData->BindToTexture(&tex);
	dim3 blockSize(Align(nK / 2, 32), 1, 1);
	dim3 gridSize(nIJ, 1, 1);
	int2 dimImg = { width, height };
	cudaBP<TRANSPOSE> << <gridSize, blockSize >> > (pDevVol->GetData(), nK, pDevIJ->GetData(), nIJ, dimImg, nProj);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR;
	return true;
}

bool BackProjection::GetVolumeData(float* pVol, int size) {
	VERIFY_TRUE(size >= nK * nIJ);
	pDevVol->CopyToHost(pVol, nK, nK, nIJ);
	return true;
}



//////////////////////////////////////////
template
bool BackProjection::BP<0>(const float* pProjData, int width, int height, float* pProjMat, int count);
template
bool BackProjection::BP<1>(const float* pProjData, int width, int height, float* pProjMat, int count);
