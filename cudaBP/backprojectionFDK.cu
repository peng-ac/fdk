#include "../common/type.h"
#include "backprojection.cuh"
#include "../common/cudaLib.cuh"
#include "cudaBpLib.cuh"


const int NP = BackProjectionFDK::MAX_PROJ; //batch count of processing proj-data   
__constant__ float4 ProjMat[NP][3]; //NPx3x4 proj-matrix
									//textures for proj-data sequence
									//texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
									//texture<float, cudaTextureType2DLayered> tex(false, cudaFilterModeLinear, cudaAddressModeClamp);
									//texture<float, cudaTextureType3D> tex(false, cudaFilterModePoint, cudaAddressModeClamp);
									//texture tex_ij; //1-D texture for (i,j) index array

__device__ __forceinline__ float SUBPIXEL(const float* pProj, int nProjPitch, int width, int height, float u, float v)
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
	}
	else {
		s = 0;
	}

	return s;
}

template<int TRANSPOSE, bool bDualBuffer, bool bSymmetric>
__global__ void cudaFdkBP(
	float* vol, int3 minA, int3 minB, 
	int3 sub_vol_dim, int3 vol_dim, 
	const float* pProj, int nProjPitch, int3 img_dim)
{
	assert(!bDualBuffer);
	//assert(bSymmetric);
	assert(TRANSPOSE);
	const uint laneId = threadIdx.x;
	uint ii = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint jj = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint kk = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

	if (bSymmetric){
		if (ii >= sub_vol_dim.z/2 || jj >= sub_vol_dim.y || kk >= sub_vol_dim.x)
			return;
	}

	uint i = ii + minA.z;
	uint j = jj + minA.y;
	uint k = kk + minA.x;

	//int _k = nK - 1 - k;  //geometry symmetric
	float4 IJKW = make_float4(k, j, i, 1.f);
	const int nProj = img_dim.z;
	float Z, U;

	assert(nProj <= 32);
	//in parallel compute U, F, F2
	if (laneId < nProj) {
		//use register
		Z = 1.f / dotIJKW(ProjMat[laneId][2], IJKW); //inner product
		U = dotIJKW(ProjMat[laneId][0], IJKW)*Z;
	}

	//in-register accumulator,
	float sum  = 0, _sum = 0;
	//geometry symmetric
	float v, _v;
	int _i;
	if (bSymmetric) _i = vol_dim.z - 1 - i;

	const int nFrameBufferSize = nProjPitch * img_dim.y;
	const float* _pProj = pProj;
	//#pragma unroll (16)
	for (int s = 0; s<nProj; s++) {
		float u = __my_shfl(U, s);
		float f = __my_shfl(Z, s);
		float f2 = f * f;
		v = dotIJKW(ProjMat[s][1], IJKW)*f;
		if (TRANSPOSE == 0) {
			if (bSymmetric)
				_v = img_dim.y - 1 - v;       //geometry symmetric	
			sum = FMAD(SUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, u, v), f2, sum);
			if (bSymmetric)
				_sum = FMAD(SUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, u, _v), f2, _sum);
		}
		if (TRANSPOSE == 1) {
			if (bSymmetric) 
				_v = img_dim.x - 1 - v;      //geometry symmetric
			sum = FMAD(SUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, v, u), f2, sum);
			if (bSymmetric) 
				_sum = FMAD(SUBPIXEL(_pProj, nProjPitch, img_dim.x, img_dim.y, _v, u), f2, _sum);
		}
		_pProj += nFrameBufferSize;
	}
	//update volume
	//uint offset = k*vol_dim.z*vol_dim.y + j*vol_dim.z;
	uint offset = (kk * sub_vol_dim.y + jj) * sub_vol_dim.z;
	float* pVolOut = vol + offset;
	{
		pVolOut[ii] += sum;		
		if (bSymmetric) {
			int _ii = sub_vol_dim.z - 1 - ii;
			pVolOut[_ii] += _sum;
		}
		//pVolOut[_k] += _sum;
	}
}

__global__ void kernelTransformVolume(const float* __restrict__ src, float* dst, uint nx, uint ny, uint nz, uint _nxx) {
	uint k = blockIdx.x*blockDim.x + threadIdx.x;
	uint j = blockIdx.y*blockDim.y + threadIdx.y;
	uint i = blockIdx.z*blockDim.z + threadIdx.z;
	uint idx = nx*(ny*i + j)+ k;
	PtValue3D(dst, nz, ny, nx, i, j, k) = src[idx];
}

void TransformVolume(const float* src, float* dst, uint _nx, uint _ny, uint _nz, uint _nxx) {
	dim3 blocks(4, 32, 4);
	dim3 grids(UpDivide(_nxx, blocks.x), UpDivide(_ny, blocks.y), UpDivide(_nz, blocks.z));
	VERIFY_TRUE(_nx%blocks.x == 0);
	VERIFY_TRUE(_ny%blocks.y == 0);
	VERIFY_TRUE(_nz%blocks.z == 0);
	kernelTransformVolume << <grids, blocks >> > (src, dst, _nx, _ny, _nz, _nxx);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR;
}

///////////////////////////////////////////////////
BackProjectionFDK::BackProjectionFDK(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, bool _bDualBuffer/* = false*/)
{
	DISPLAY_FUNCTION;
	VERIFY_TRUE(0);
}

BackProjectionFDK::~BackProjectionFDK() {
	if (pVolBuffer) cudaFree(pVolBuffer);
}

BackProjectionFDK::BackProjectionFDK(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, geo::BoxT<size_t> _subTop, geo::BoxT<size_t> _subBottom, bool _bDualBuffer /*= false*/){
	//DISPLAY_FUNCTION;
	VERIFY_TRUE(!bDualBuffer);
	pVolBuffer = NULL;
	nBpCount = 0;
	//std::cout << StringFormat("########################subTop.MinZ()=%d, subBottom.MaxZ()=%d, _nz =%d\n", subTop.MinZ(), subBottom.MaxZ(), _nz);
	subTop = _subTop;
	subBottom = _subBottom;
	//VERIFY_TRUE(subTop.MinX() + subBottom.MaxX() == _nx - 1);
	//VERIFY_TRUE(subTop.MinY() + subBottom.MaxY() == _ny - 1);
	if (subBottom.depth > 0) {
		VERIFY_TRUE(subTop.MinZ() + subBottom.MaxZ() == _nz - 1);
	}
	//std::cout << "MAX_PROJ = " << MAX_PROJ << std::endl;
	int tm = timeGetTime();
	nx = _nx; ny = _ny; nz = _nz;
	projWidth = _projWidth; projHeight = _projHeight;  projCount = _projCount;
	bDualBuffer = _bDualBuffer;

	pDevProjData = std::make_shared<GpuBuffer<float>>(_projWidth, _projHeight, _projCount);
	//pDevProjData->Zero();
	pDevVolOut = std::make_shared<GpuBuffer<float>>(subTop.depth+subBottom.depth, subTop.height, subTop.width);
	pDevVolOut->Zero();
	if (bDualBuffer) {
		VERIFY_TRUE(0);
	} else {
		pDevVolIn = pDevVolOut;
	}
	size_t totalBytes = getTotalBytes();
	int iTypeMem = 0;
	static const char sMemType[2][64]{
		"DeviceMem", "ManagedMem",
	};
#if 0
	size_t freeBytes = totalBytes - projWidth * projHeight*projCount * sizeof(float);
	size_t nBytes = (subTop.depth + subBottom.depth) * subTop.height*subTop.width * sizeof(float);
	if (freeBytes/2 > nBytes) {
		iTypeMem = 0;
		pDevData = std::make_shared<GpuBuffer<float>>(subTop.width, subTop.height, subTop.depth + subBottom.depth);
	} else 
	{
		iTypeMem = 1;
		printf("cudaMallocManaged : %f GB\n", ToGB(sizeof(*pVolBuffer)*(subTop.depth + subBottom.depth) * subTop.height*subTop.width));
		cudaMallocManaged(&pVolBuffer, sizeof(*pVolBuffer)*(subTop.depth + subBottom.depth) * subTop.height*subTop.width);
		VERIFY_TRUE(pVolBuffer);
	}
#else
	size_t freeBytes = getFreeBytes();
	size_t nDep = subTop.depth + subBottom.depth;
	while (freeBytes < subTop.height*subTop.width*nDep*sizeof(float)) nDep /= 2;
	//for debug
	//nDep /= 2;

	pDevData = std::make_shared<GpuBuffer<float>>(subTop.width, subTop.height, nDep);
#endif

	//tmpVol = std::make_shared<VolumeT<float>>(subTop.depth * 2, subTop.height, subTop.width);
	CUDA_CHECK_ERROR;
	tm = timeGetTime() - tm;
	int id;
	cudaGetDevice(&id);
	printf("%s:gpu(%d),MemSize=%f Gb, Use %s (nDep=%d <--%d), MAX_PROJ = %d, proj:(%d-%d-%d), DualBuffer = %s, init time = %d ms\n", 
		hostname().c_str(), id,
		ToGB(totalBytes), sMemType[iTypeMem], nDep, subTop.depth + subBottom.depth, MAX_PROJ, _projWidth, _projHeight, _projCount, bDualBuffer ? "true" : "false", tm);
	//printf("DualBuffer = %s\n", bDualBuffer ? "true" : "false");
	//std::cout << "init time = " << tm << " ms" << std::endl;
}

bool BackProjectionFDK::BP(int TRANSPOSE, const float* pProjData, int width, int height, float* pProjMat, int nProj) {
	const bool bSymmetric = subBottom.depth == 0?false:true;
	bool bRtn = false;
	if (0) {}
	else if (TRANSPOSE == 0 && bSymmetric == false) bRtn = inBP<0, 0>(pProjData, width, height, pProjMat, nProj);
	else if (TRANSPOSE == 0 && bSymmetric == true)  bRtn = inBP<0, 1>(pProjData, width, height, pProjMat, nProj);
	else if (TRANSPOSE == 1 && bSymmetric == false) bRtn = inBP<1, 0>(pProjData, width, height, pProjMat, nProj);
	else if (TRANSPOSE == 1 && bSymmetric == true)  bRtn = inBP<1, 1>(pProjData, width, height, pProjMat, nProj);
	return bRtn;
}

template<int TRANSPOSE, bool bSymmetric>
bool BackProjectionFDK::inBP(const float* pProjData, int width, int height, float* pProjMat, int nProj) {
	nBpCount += nProj;
	//VERIFY_TRUE(cudaSuccess == cudaFuncSetCacheConfig(cudaBP<TRANSPOSE>, cudaFuncCachePreferL1));
	if (bDualBuffer) SwapInOutBuffer();
	LoadConstant(&ProjMat[0][0], pProjMat, PROJ_MAT_SIZE*nProj);
	pDevProjData->CopyFromHost(pProjData, width, width, height, nProj);
	//pDevProjData->BindToTexture(&tex);
#ifdef _DEBUG 
	dim3 blockSize(WarpSize, 1, 1);
#else
	dim3 blockSize(WarpSize, 8, 4);
#endif
	dim3 gridSize(UpDivide(subTop.depth, blockSize.x), UpDivide(subTop.height, blockSize.y), UpDivide(subTop.width, blockSize.z));
	//if (!bSymmetric) gridSize.x = UpDivide(nz, blockSize.x);
	int3 vol_dim = {nx, ny, nz};
	int3 dimImg = { width, height, nProj };
	/*if (bDualBuffer)
		cudaBP<TRANSPOSE, true, bSymmetric> << <gridSize, blockSize >> > (pDevVolIn->GetData(), pDevVolOut->GetData(), vol_dim, pDevProjData->GetData(), pDevProjData->DataPitch(), dimImg);
	else */
	{
		float* vol = pDevVolOut->GetData();
		int3 minA = { subTop.MinX(), subTop.MinY(), subTop.MinZ() };
		int3 maxA = { subTop.MaxX(), subTop.MaxY(), subTop.MaxZ() };
		int3 minB = { subBottom.MinX(), subBottom.MinY(), subBottom.MinZ() };
		int3 maxB = { subBottom.MaxX(), subBottom.MaxY(), subBottom.MaxZ() };
		int3 sub_vol_dim = { subTop.width, subTop.height, subTop.depth + subBottom.depth };
		int3 vol_dim = { nx, ny, nz };
		cudaFdkBP<TRANSPOSE, false, bSymmetric> << <gridSize, blockSize >> > (
			vol, minA, minB,
			sub_vol_dim, vol_dim,
			pDevProjData->GetData(), pDevProjData->DataPitch(), dimImg);
	}
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR;
	return true;
}

bool BackProjectionFDK::GetVolumeData(float* pVol, int size) {
	//VERIFY_TRUE(size >= nx*ny*nz);
	//VolumeT<float> vol(subTop.depth * 2, subTop.height, subTop.width);
#if 1
	if (pDevData) {
		printf("GetVolumeData by DeviceMem\n");
		int tmTrans = 0;
		int tmCopy = 0;
		for (size_t s = 0; s < subTop.depth + subBottom.depth; s += pDevData->depth) {
			VERIFY_TRUE(pDevData->depth <= (subTop.depth + subBottom.depth));
			VERIFY_TRUE((subTop.depth + subBottom.depth)% pDevData->depth == 0);
			int tm0 = timeGetTime();
			TransformVolume(pDevVolOut->data+s, pDevData->data, subTop.depth + subBottom.depth, subTop.height, subTop.width, pDevData->depth);
			int tm1 = timeGetTime();
			pDevData->CopyToHost(pVol + subTop.width*subTop.height*s, subTop.width, subTop.width, subTop.height, pDevData->depth);
			int tm2 = timeGetTime();
			tmTrans += tm1 - tm0;
			tmCopy += tm2 - tm1;
		}
		//int tm0 = timeGetTime();
		//TransformVolume(pDevVolOut->data, pDevData->data, subTop.depth + subBottom.depth, subTop.height, subTop.width);
		//int tm1 = timeGetTime();
		//pDevData->CopyToHost(pVol, subTop.width, subTop.width, subTop.height, subTop.depth + subBottom.depth);
		//int tm2 = timeGetTime();
		printf("transform-time=%d ms, CopyToHost-time=%d ms, by DeviceMem\n", tmTrans, tmCopy);
	} 
	//else {
	//	printf("GetVolumeData by ManagedMem\n");
	//	VERIFY_TRUE(pVolBuffer);
	//	int tm0 = timeGetTime();
	//	TransformVolume(pDevVolOut->data, pVolBuffer, subTop.depth + subBottom.depth, subTop.height, subTop.width);
	//	int tm1 = timeGetTime();
	//	memcpy(pVol, pVolBuffer, size * sizeof(*pVol));
	//	int tm2 = timeGetTime();
	//	printf("transform-time=%d ms, CopyToHost-time=%d ms, by ManagedMem\n", tm1 - tm0, tm2 - tm1);
	//}
#if 0
#else
#endif
#else
	int _nx = subTop.depth * 2;
	int _ny = subTop.height;
	int _nz = subTop.width;
	VolumeT<float> vol(_nx, _ny, _nz);
	int tm = timeGetTime();
	pDevVolOut->CopyToHost(vol.buffer, subTop.depth*2, subTop.depth * 2, subTop.height, subTop.width);
	tm = timeGetTime() - tm;
	printf("pDevVolOut->CopyToHost time=%d ms\n", tm);

	tm = timeGetTime();
#if 0
	//int idx = 0;
	#pragma omp parallel for num_threads(4)	
	for (int i = 0; i < vol.nz; i++) {
		for (int j = 0; j < vol.ny; j++) {
			for (int k = 0; k < vol.nx; k++, idx++) {
				int idx = vol.nx*vol.ny*i+j*vol.nx+k;
				PtValue3D(pVol, vol.nz, vol.ny, vol.nx, i, j, k) = vol.buffer[idx];
			}
		}
	}
#else
	//subTop.depth * 2, subTop.height, subTop.width
	int nxy = _nx*_ny;
	int nxyz = nxy * _nz;
	int i = 0, j = 0, k = 0;

	#pragma omp parallel for schedule(static,1) num_threads(32)
	for (int idx = 0; idx < nxyz; idx++) {
		int k = idx % _nx;
		int j = idx % nxy / _nx;
		int i = idx / nxy;
		//VERIFY_TRUE(i < vol.nz);
		//VERIFY_TRUE(j < vol.ny);
		//VERIFY_TRUE(k < vol.nx);

		PtValue3D(pVol, vol.nz, vol.ny, vol.nx, i, j, k) = vol.buffer[idx];
	}	
	tm = timeGetTime() - tm;
	printf("transpose time=%d ms\n", tm);
#endif
#endif

	return true;
}

inline void BackProjectionFDK::SwapInOutBuffer() {
	VERIFY_TRUE(pDevVolIn);
	auto tmp = this->pDevVolIn;
	pDevVolIn = pDevVolOut;
	pDevVolOut = tmp;
}


////////////////////////////////////////////
//template
//bool BackProjectionFDK::inBP<0>(const float* pProjData, int width, int height, float* pProjMat, int count);
//template
//bool BackProjectionFDK::inBP<1>(const float* pProjData, int width, int height, float* pProjMat, int count);
