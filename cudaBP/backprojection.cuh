#ifndef __CUDABP_H
#define __CUDABP_H
#include <memory>
#include <vector>
#include <assert.h>

template<typename T> struct GpuData;
template<class T> struct GpuArray3D;
template<class T> struct GpuBuffer;
struct float2;
struct cudaArray;

#ifndef USE_8BITS
#define USE_8BITS 0
	#if USE_8BITS == 1
		#pragma message "Use 8Bits Precision"
	#elif USE_8BITS == 0
		#pragma message "Use 32Bits Precision"
	#endif
#endif

namespace geo {
	template<typename T>
	struct Point3_T {
		Point3_T() :x(0), y(0), z(0) {
		}
		template<typename _Tp0, typename _Tp1, typename _Tp2>
		Point3_T(_Tp0 _x, _Tp1 _y, _Tp2 _z) : x(_x), y(_y), z(_z) {
		}
		Point3_T& operator=(const Point3_T& pt) {
			x = pt.x;
			y = pt.y;
			z = pt.z;
			return *this;
		}
		Point3_T& operator+=(const Point3_T& pt) {
			x += pt.x;
			y += pt.y;
			z += pt.z;
			return *this;
		}
		bool operator ==(const Point3_T& pt) const {
			return (x == pt.x && y == pt.y && z == pt.z) ? true : false;
		}
		T x, y, z;
	};

	template<typename _Tp>
	struct Size3_T {
		Size3_T(_Tp _cx = 0, _Tp _cy = 0, _Tp _cz = 0) :cx(_cx), cy(_cy), cz(_cz) {
		}
		Size3_T(const Size3_T& sz) {
			*this = sz;
		}
		inline Size3_T& operator=(const Size3_T& sz) {
			cx = sz.cx;
			cy = sz.cy;
			cz = sz.cz;
			return *this;
		}
		inline bool operator==(const Size3_T& sz) const {
			return (sz.cx == cx && sz.cy == cy && sz.cz == cz) ? true : false;
		}
		_Tp cx, cy, cz;
	};

	template<typename _Tp>
	struct BoxT {
		typedef _Tp DataType;
		BoxT() {
		}
		BoxT(_Tp _x, _Tp _y, _Tp _z, _Tp _width, _Tp _height, _Tp _depth) {
			x = _x;
			y = _y;
			z = _z;
			width = _width;
			height = _height;
			depth = _depth;
		}
		inline BoxT& operator=(const BoxT& box) {
			x = box.x;
			y = box.y;
			z = box.z;
			width = box.width;
			height = box.height;
			depth = box.depth;
			return *this;
		}
		template<typename T>
		inline bool IsInside(Point3_T<T> pt) const {
			return pt.x >= x && pt.x <= x + width - 1
				&& pt.y >= y && pt.y <= y + height - 1
				&& pt.z >= z && pt.z <= z + depth - 1;
		}
		Size3_T<_Tp> Size() const {
			return Size3_T<_Tp>(width, height, depth);
		}
		BoxT& Offset(_Tp _x, _Tp _y, _Tp _z) {
			x += _x; y += _y; z += _z;
			return *this;
		}
		BoxT& Offset(Point3_T<_Tp> offset) {
			return this->Offset(offset.x, offset.y, offset.z);
		}
		bool IsValid() const {
			return (width > 0 && height > 0 && depth > 0) ? true : false;
		}
		_Tp MinX() { return x; }
		_Tp MinY() { return y; }
		_Tp MinZ() { return z; }
		_Tp MaxX() { return x + width - 1; }
		_Tp MaxY() { return y + height - 1; }
		_Tp MaxZ() { return z + depth - 1; }

		_Tp x, y, z, width, height, depth;
	};
};

struct cudaBackProjection {
	enum {
		MAX_PROJ = 32,
		PROJ_MAT_ROWS = 3,
		PROJ_MAT_COLS = 4,
		PROJ_MAT_SIZE = PROJ_MAT_ROWS * PROJ_MAT_COLS,
	};
	cudaBackProjection() {
		bDualBuffer = false;
		fKernelTime = 0;
	}
	float fKernelTime;
	bool bDualBuffer;
};

struct BackProjectionTexture : public cudaBackProjection {
public:
	BackProjectionTexture(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, bool bDualBuffer = false) {
		assert(0);
	}
	BackProjectionTexture(int _nK, int _projWidth, int _projHeight, int _projCount, const int* pIJ, int _nIJ, bool _bDualBuffer = true);
	virtual ~BackProjectionTexture();
	bool GetVolumeData(float* pVol, int size);
	bool BP(int TRANSPOSE, const float* pProjData, int width, int height, float* pProjMat, int nProj);
private:
	template<int TRANSPOSE>
	bool inBP(const float* pProjData, int width, int height, float* pProjMat, int nProj);
public:
	std::shared_ptr<GpuData<float2>> pDevIJ;
	std::shared_ptr<GpuData<float>> pDevVolIn, pDevVolOut;
	std::shared_ptr<GpuArray3D<float>> pDevProjData;
	std::vector<int> vecIJ;
	std::vector<float> vecfIJ;
	int projWidth, projHeight, projCount, nK, nIJ;
private:
	void SwapInOutBuffer();
};

struct BackProjectionGlobalmem : public cudaBackProjection {
public:
	BackProjectionGlobalmem(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, bool bDualBuffer = false) {
		assert(0);
	}
	BackProjectionGlobalmem(int _nK, int _projWidth, int _projHeight, int _projCount, const int* pIJ, int _nIJ, bool _bDualBuffer = true);
	virtual ~BackProjectionGlobalmem();
	bool GetVolumeData(float* pVol, int size);
	bool BP(int TRANSPOSE, const float* pProjData, int width, int height, float* pProjMat, int nProj);
private:
	template<int TRANSPOSE>
	bool inBP(const float* pProjData, int width, int height, float* pProjMat, int nProj);
public:
	std::shared_ptr<GpuData<float2>> pDevIJ;
	std::shared_ptr<GpuData<float>> pDevVolIn, pDevVolOut;
	std::shared_ptr<GpuData<float>> pDevProjData;
	std::vector<int> vecIJ;
	std::vector<float> vecfIJ;
	int projWidth, projHeight, projCount, nK, nIJ;
private:
	void SwapInOutBuffer();
};

struct BackProjectionRTK : public cudaBackProjection {
public:
	BackProjectionRTK(int _nK, int _projWidth, int _projHeight, int _projCount, const int* pIJ, int _nIJ, bool bDualBuffer = false) {
		assert(0);
	}
	BackProjectionRTK(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, bool bDualBuffer = false);
	virtual ~BackProjectionRTK();
	bool BP(int TRANSPOSE, float* pProjData, int width, int height, float* pProjMat, int nProj);
	bool GetVolumeData(float* pVol, int size);
public:
	std::shared_ptr<GpuBuffer<float>> pDevVolIn, pDevVolOut;
	//std::shared_ptr<GpuArray3D<float>> pDevProjData;
	int projWidth, projHeight, projCount;
	int nx, ny, nz;
private:
	cudaArray * array_proj;
	void CUDA_reconstruct_conebeam(int proj_size[3], int vol_size[3], float *matrices, float *dev_vol_in, float *dev_vol_out, float *dev_proj);
	void SwapInOutBuffer();
};

struct BackProjectionRevRTK : public cudaBackProjection {
public:
	BackProjectionRevRTK(int _nK, int _projWidth, int _projHeight, int _projCount, const int* pIJ, int _nIJ, bool bDualBuffer = false) {
		assert(0);
	}
	BackProjectionRevRTK(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, bool bDualBuffer = false);
	virtual ~BackProjectionRevRTK();
	bool BP(int TRANSPOSE, const float* pProjData, int width, int height, float* pProjMat, int nProj);
	bool GetVolumeData(float* pVol, int size);
private:
	template<int TRANSPOSE, bool bSymmetric>
	bool inBP(const float* pProjData, int width, int height, float* pProjMat, int nProj);
public:
	std::shared_ptr<GpuBuffer<float>> pDevVolIn, pDevVolOut;
	std::shared_ptr<GpuBuffer<float>> pDevProjData;
	int projWidth, projHeight, projCount;
	int nx, ny, nz;
private:
	cudaArray * array_proj;

	void SwapInOutBuffer();
};

struct BackProjectionTextureRevRTK : public cudaBackProjection {
public:
	BackProjectionTextureRevRTK(int _nK, int _projWidth, int _projHeight, int _projCount, const int* pIJ, int _nIJ, bool bDualBuffer = false) {
		assert(0);
	}
	BackProjectionTextureRevRTK(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, bool bDualBuffer = false);
	virtual ~BackProjectionTextureRevRTK();
	bool BP(int TRANSPOSE, const float* pProjData, int width, int height, float* pProjMat, int nProj);
	bool GetVolumeData(float* pVol, int size);
private:
	template<int TRANSPOSE, bool bSymmetric>
		bool inBP(const float* pProjData, int width, int height, float* pProjMat, int nProj);
public:
	std::shared_ptr<GpuBuffer<float>> pDevVolIn, pDevVolOut;
	std::shared_ptr<GpuArray3D<float>> pDevProjData;
	int projWidth, projHeight, projCount;
	int nx, ny, nz;
private:
	cudaArray * array_proj;

	void SwapInOutBuffer();
};

struct BackProjectionFDK : public cudaBackProjection {
public:
	BackProjectionFDK(int _nK, int _projWidth, int _projHeight, int _projCount, const int* pIJ, int _nIJ, bool bDualBuffer = false) {
		assert(0);
	}
	BackProjectionFDK(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, bool bDualBuffer = false);
	BackProjectionFDK(int _projWidth, int _projHeight, int _projCount, int _nx, int _ny, int _nz, geo::BoxT<size_t> _subTop, geo::BoxT<size_t> _subBottom, bool bDualBuffer = false);
	virtual ~BackProjectionFDK();
	bool BP(int TRANSPOSE, const float* pProjData, int width, int height, float* pProjMat, int nProj);
	bool GetVolumeData(float* pVol, int size);
private:
	template<int TRANSPOSE, bool bSymmetric>
	bool inBP(const float* pProjData, int width, int height, float* pProjMat, int nProj);
public:
	std::shared_ptr<GpuBuffer<float>> pDevVolIn, pDevVolOut;
	std::shared_ptr<GpuBuffer<float>> pDevProjData;
	size_t projWidth, projHeight, projCount;
	size_t nx, ny, nz;
	geo::BoxT<size_t> subTop, subBottom;
	int nBpCount;
private:
	cudaArray * array_proj;
	float* pVolBuffer;
	std::shared_ptr<GpuBuffer<float>> pDevData;
	void SwapInOutBuffer();
};

#endif