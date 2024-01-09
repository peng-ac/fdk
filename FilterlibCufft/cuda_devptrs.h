#ifndef __DEVPTRS_H__
#define __DEVPTRS_H__

#include "device_launch_parameters.h"

#ifdef __cplusplus

#define __GPU_HOST_DEVICE__ __host__ __device__ __forceinline__


// Simple lightweight structures that encapsulates information about an image on device.
// It is intended to pass to nvcc-compiled code. GpuMat depends on headers that nvcc can't compile

template <bool expr> struct StaticAssert;
template <> struct StaticAssert<true> {static __GPU_HOST_DEVICE__ void check(){}};

template<typename T> struct DevPtr
{
	typedef T elem_type;
	typedef int index_type;

	enum { elem_size = sizeof(elem_type) };

	T* data;

	__GPU_HOST_DEVICE__ DevPtr() : data(0) {}
	__GPU_HOST_DEVICE__ DevPtr(T* data_) : data(data_) {}

	__GPU_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
	__GPU_HOST_DEVICE__ operator       T*()       { return data; }
	__GPU_HOST_DEVICE__ operator const T*() const { return data; }
};

template<typename T> struct PtrSz : public DevPtr<T>
{
	__GPU_HOST_DEVICE__ PtrSz() : size(0) {}
	__GPU_HOST_DEVICE__ PtrSz(T* data_, size_t size_) : DevPtr<T>(data_), size(size_) {}

	size_t size;
};

template<typename T> struct PtrStep : public DevPtr<T>
{
	__GPU_HOST_DEVICE__ PtrStep() : step(0) {}
	__GPU_HOST_DEVICE__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}

	/** \brief stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!! */
	size_t step;

	__GPU_HOST_DEVICE__       T* ptr(int y = 0)       { return (      T*)( (      char*)DevPtr<T>::data + y * step); }
	__GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( (const char*)DevPtr<T>::data + y * step); }

	__GPU_HOST_DEVICE__       T& operator ()(int y, int x)       { return ptr(y)[x]; }
	__GPU_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
};

template <typename T> struct PtrStepSz : public PtrStep<T>
{
	__GPU_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
	__GPU_HOST_DEVICE__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_)
		: PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}

	template <typename U>
	explicit PtrStepSz(const PtrStepSz<U>& d) : PtrStep<T>((T*)d.data, d.step), cols(d.cols), rows(d.rows){}

	int cols;
	int rows;
};

typedef PtrStepSz<unsigned char> PtrStepSzb;
typedef PtrStepSz<float> PtrStepSzf;
typedef PtrStepSz<int> PtrStepSzi;

typedef PtrStep<unsigned char> PtrStepb;
typedef PtrStep<float> PtrStepf;
typedef PtrStep<int> PtrStepi;



template <typename T> struct DevMem2D_ : public PtrStepSz<T>
{
	DevMem2D_() {}
	DevMem2D_(int rows_, int cols_, T* data_, size_t step_) : PtrStepSz<T>(rows_, cols_, data_, step_) {}

	template <typename U>
	explicit  DevMem2D_(const DevMem2D_<U>& d) : PtrStepSz<T>(d.rows, d.cols, (T*)d.data, d.step) {}
};

typedef DevMem2D_<unsigned char> DevMem2Db;
typedef DevMem2Db DevMem2D;
typedef DevMem2D_<float> DevMem2Df;
typedef DevMem2D_<int> DevMem2Di;

template<typename T> struct PtrElemStep_ : public PtrStep<T>
{
	PtrElemStep_(const DevMem2D_<T>& mem) : PtrStep<T>(mem.data, mem.step)
	{
		StaticAssert<256 % sizeof(T) == 0>::check();

		PtrStep<T>::step /= PtrStep<T>::elem_size;
	}
	__GPU_HOST_DEVICE__ T* ptr(int y = 0) { return PtrStep<T>::data + y * PtrStep<T>::step; }
	__GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return PtrStep<T>::data + y * PtrStep<T>::step; }

	__GPU_HOST_DEVICE__ T& operator ()(int y, int x) { return ptr(y)[x]; }
	__GPU_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
};

template<typename T> struct PtrStep_ : public PtrStep<T>
{
	PtrStep_() {}
	PtrStep_(const DevMem2D_<T>& mem) : PtrStep<T>(mem.data, mem.step) {}
};

typedef PtrElemStep_<unsigned char> PtrElemStep;
typedef PtrElemStep_<float> PtrElemStepf;
typedef PtrElemStep_<int> PtrElemStepi;




#endif // __cplusplus

#endif /* __DEVPTRS_H__ */
