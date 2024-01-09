#ifndef __cudaLib_H
#define __cudaLib_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string.h>
#include <stdarg.h>
#ifdef _WIN32
#include <Windows.h>
#include <winsock.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <unistd.h>
#endif
#include <assert.h>

inline int GetDeviceID() {
	int device = -1;
	if (cudaGetDevice(&device) != cudaSuccess)
		device = -1;
	return device;
}

inline std::string CUDAHostname() {
	char szBuf[4096] = "";
	gethostname(szBuf, sizeof(szBuf));
	return szBuf;
}

inline std::string CUDAString(const char* format, ...) {
	char szBuf[4096] = "";
	va_list arg;

	va_start(arg, format);
	vsnprintf(szBuf, sizeof(szBuf), format, arg);
	va_end(arg);
	std::string str = szBuf;
	sprintf(szBuf, "%s:gpuId=%d:%s", CUDAHostname().c_str(), GetDeviceID(), str.c_str());
	return std::string(szBuf);
}

#define __cudaCheckError(x)  if (cudaSuccess != (x)){          \
	printf("file %s, line %d, function %s,\nCUDA ERROR:%s\n",  \
	__FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(x));  \
	assert(0);throw(std::string(cudaGetErrorString(x)));       \
}

#define __cudaSafeCall(x)    if (cudaSuccess != (x)){__cudaCheckError((x));}
#define CUDA_CHECK_ERROR { cudaError_t err = cudaGetLastError(); __cudaCheckError(err); }

#define _cudaCheckError(x, _msg)  if (cudaSuccess != (x)){           \
	printf("file %s, line %d, function %s,\nCUDA ERROR:%s\n details:%s\n",  \
	__FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(x), _msg.c_str());  \
	assert(0);throw(std::string(cudaGetErrorString(x)) + "\n details:" + _msg);  \
}
#define _CUDA_CHECK_ERROR(_msg) { cudaError_t err = cudaGetLastError(); _cudaCheckError(err, std::string(_msg));}

//convert const pointer to general one
template<typename T> inline
T* ConstToGeneral(const T* ptr) { T* _ptr; memcpy(&_ptr, &ptr, sizeof(_ptr)); return _ptr; }

#define WarpSize 32
__device__ __host__ __forceinline__ unsigned int Align(unsigned int x, unsigned int y) { assert(y != 0); return (x + y - 1) / y*y; }
__device__ __host__ __forceinline__ unsigned int UpDivide(unsigned int x, unsigned int y) { assert(y != 0); return (x + y - 1) / y; }
__device__ __host__ __forceinline__ unsigned int UpRound(unsigned int x, unsigned int y) { return UpDivide(x, y)*y; }

struct GpuStream {
	GpuStream() :stream(cudaStreamDefault) {
	}
	~GpuStream() {
		Destroy();
	}
	inline GpuStream& Create() {
		Destroy();
		cudaStreamCreate(&stream);
		CUDA_CHECK_ERROR;
		return *this;
	}
	inline void Destroy() {
		if (!IsDefault()) cudaStreamDestroy(stream);
	}
	inline bool IsDefault() const {
		return stream == cudaStreamDefault ? true : false;
	}
	cudaStream_t stream;
};


template<class T, int DIM>
struct GpuDataPtr {
	T* data;
	size_t width, data_pitch, height, depth;
};


#define TYPE_GpuDataPtr1(type)\
	template<> struct GpuDataPtr<type, 1>{\
	__device__ __host__ __forceinline__ GpuDataPtr(type* _data, size_t _width):data(_data), width(_width){}\
	__device__ __host__ __forceinline__ float& operator()(size_t x) const{\
		return data[x];\
	}\
	__device__ __host__ __forceinline__ GpuDataPtr& operator=(const GpuDataPtr& obj){\
		data = obj.data;width = obj.width;\
		return *this;\
	}\
	type* data;\
	size_t width;\
};

#define TYPE_GpuDataPtr2(type)\
	template<>struct GpuDataPtr<type, 2>{\
	__device__ __host__ __forceinline__ GpuDataPtr(type* _data, size_t _width, size_t _data_pitch, size_t _height)\
		:data(_data), width(_width), data_pitch(_data_pitch), height(_height){\
	}\
	__device__ __host__ __forceinline__ float& operator()(size_t x, size_t y) const{\
		return data[x + y*data_pitch];\
	}\
	template<typename _Tp>\
	__device__ __host__ __forceinline__ bool IsValid(_Tp x, _Tp y) const{\
		return x>=0 && x<=width-1 && y>=0 && y<=height-1;\
	}\
	__device__ __host__ __forceinline__ GpuDataPtr& operator=(const GpuDataPtr& obj){\
		data = obj.data;width = obj.width; height = obj.height;\
		return *this;\
	}\
	type* data;\
	size_t width, data_pitch, height;\
};

#define TYPE_GpuDataPtr3(type)\
	template<>struct GpuDataPtr<type, 3>{\
	__device__ __host__ __forceinline__ GpuDataPtr(type* _data, size_t _width, size_t _data_pitch, size_t _height, size_t _depth)\
		:data(_data), width(_width), data_pitch(_data_pitch), height(_height), depth(_depth){\
	}\
	__device__ __host__ __forceinline__ float& operator()(size_t x, size_t y, size_t z) const{\
		return data[x + (y + z*height)*data_pitch];\
	}\
	template<typename _Tp>\
	__device__ __host__ __forceinline__ bool IsValid(_Tp x, _Tp y, _Tp z) const{\
		return x>=0 && x<=width-1 && y>=0 && y<=height-1 && z>=0 && z<=depth-1;\
	}\
	__device__ __host__ __forceinline__ GpuDataPtr& operator=(const GpuDataPtr& obj){\
		data = obj.data;width = obj.width; height = obj.height; depth = obj.depth;\
		return *this;\
	}\
	type* data;\
	size_t width, data_pitch, height, depth;\
};

TYPE_GpuDataPtr1(float);
TYPE_GpuDataPtr2(float);
TYPE_GpuDataPtr3(float);


//	template<>struct GpuDataPtr<float, 2>{\
//	__device__ __host__ __forceinline__ __forceinline__ GpuDataPtr(float* _data, size_t _width, size_t _data_pitch, size_t _height)\
//		:data(_data), width(_width), data_pitch(_data_pitch), height(_height){\
//	}\
//	__device__ __host__ __forceinline__ __forceinline__ float& operator()(size_t x, size_t y) const{\
//		return data[x + y*data_pitch];\
//	}\
//	template<typename _Tp>\
//	__device__ __host__ __forceinline__ __forceinline__ bool IsValid(_Tp x, _Tp y) const{\
//		return x>=size_t(0) && x<=width-1 && y>=size_t(0) && y<=height-1;\
//	}\
//	__device__ __host__ __forceinline__ __forceinline__ GpuDataPtr& operator=(const GpuDataPtr& obj){\
//		data = obj.data;width = obj.width; height = obj.height;\
//		return *this;\
//	}\
//	float* data;\
//	size_t width, data_pitch, height;\
//};


//template<>struct GpuDataPtr<float, 2>{
//	__device__ __host__ __forceinline__ GpuDataPtr(float* _data, size_t _width, size_t _data_pitch, size_t _height)
//		:data(_data), width(_width), data_pitch(_data_pitch), height(_height){
//	}
//	__device__ __host__ __forceinline__ float& operator()(int x, int y) const{
//		return data[x + y*data_pitch];
//	}
//	__device__ __host__ __forceinline__ GpuDataPtr& operator=(const GpuDataPtr& obj){
//		data = obj.data;
//		width = obj.width; height = obj.height;
//		return *this;
//	}
//	float* data;
//	size_t width, data_pitch, height;
//};
//
//template<>struct GpuDataPtr<float, 3>{
//	__device__ __host__ __forceinline__ GpuDataPtr(float* _data, size_t _width, size_t _data_pitch, size_t _height, size_t _depth)
//		:data(_data), width(_width), data_pitch(_data_pitch), height(_height), depth(_depth){
//	}
//	__device__ __host__ __forceinline__ float& operator()(int x, int y, int z) const{
//		return data[x + y*data_pitch + z*data_pitch*height];
//	}
//	__device__ __host__ __forceinline__ GpuDataPtr& operator=(const GpuDataPtr& obj){
//		data = obj.data;
//		width = obj.width; height = obj.height; depth = obj.depth;
//		return *this;
//	}
//	float* data;
//	size_t width, data_pitch, height, depth;
//};




template<class T>
struct GpuDataRef {
	typedef GpuDataPtr<T, 1> Ptr1D;
	typedef GpuDataPtr<T, 2> Ptr2D;
	typedef GpuDataPtr<T, 3> Ptr3D;

	inline Ptr1D GetPtr1D() const {
		return Ptr1D(data, width);
	}
	inline Ptr2D GetPtr2D() const {
		return Ptr2D(data, width, data_pitch, height*depth);
	}
	inline Ptr3D GetPtr3D() const {
		return Ptr3D(data, width, data_pitch, height, depth);
	}
	inline GpuDataRef() {
		width = height = depth = data_pitch;
		data = 0;
	}
	inline GpuDataRef(T* ptr, size_t w, size_t pitch, size_t h, size_t d) {
		data = ptr; width = w; height = h; depth = d; data_pitch = pitch;
	}
	inline GpuDataRef& operator=(const GpuDataRef& obj) {
		data = obj.data; width = obj.width; height = obj.height; depth = obj.depth; data_pitch = obj.data_pitch;
		return *this;
	}
	inline void Zero(cudaStream_t stream = cudaStreamDefault) {
		if (stream == cudaStreamDefault)
			cudaMemset2D(data, data_pitch * sizeof(*data), 0, width * sizeof(*data), height*depth);
		else
			cudaMemset2DAsync(data, data_pitch * sizeof(*data), 0, width * sizeof(*data), height*depth, stream);
	}
	inline void Ones(cudaStream_t stream = cudaStreamDefault) {
		SetValue(T(1), stream);
	}
	inline void SetValue(T val, cudaStream_t stream = cudaStreamDefault) {
		size_t size = width * height*depth;
		std::vector<T> vec(size + 1);
		T* buf = &vec[0];
		for (size_t i = 0; i<size; i++) buf[i] = val;
		this->CopyFromHost(buf, width, width, height, depth, stream);
	}
	inline void CopyToHost(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d, cudaStream_t stream = cudaStreamDefault) const {
		assert(_w <= width && _h*_d <= height * depth && _w <= _pitch);
		if (stream == cudaStreamDefault)
			cudaMemcpy2D(ptr, sizeof(*ptr)*_pitch, data, data_pitch * sizeof(*data), width * sizeof(*data), _h*_d, cudaMemcpyDeviceToHost);
		else
			cudaMemcpy2DAsync(ptr, sizeof(*ptr)*_pitch, data, data_pitch * sizeof(*data), width * sizeof(*data), _h*_d, cudaMemcpyDeviceToHost, stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyToDevice(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d, cudaStream_t stream = cudaStreamDefault) const {
		assert(_w <= width && _h*_d <= height * depth && _w <= _pitch);
		if (stream == cudaStreamDefault)
			cudaMemcpy2D(ptr, sizeof(*ptr)*_pitch, data, data_pitch * sizeof(*data), width * sizeof(*data), _h*_d, cudaMemcpyDeviceToDevice);
		else
			cudaMemcpy2DAsync(ptr, sizeof(*ptr)*_pitch, data, data_pitch * sizeof(*data), width * sizeof(*data), _h*_d, cudaMemcpyDeviceToDevice, stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyFromHost(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d, cudaStream_t stream = cudaStreamDefault) {
		assert(_w <= width && _h*_d <= height * depth && _w <= _pitch);
		if (stream == cudaStreamDefault)
			cudaMemcpy2D(data, data_pitch * sizeof(*data), ptr, sizeof(*ptr)*_pitch, _w * sizeof(*data), _h*_d, cudaMemcpyHostToDevice);
		else
			cudaMemcpy2DAsync(data, data_pitch * sizeof(*data), ptr, sizeof(*ptr)*_pitch, _w * sizeof(*data), _h*_d, cudaMemcpyHostToDevice, stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyFromDevice(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d, cudaStream_t stream = cudaStreamDefault) {
		assert(_w <= width && _h*_d <= height * depth && _w <= _pitch);
		if (stream == cudaStreamDefault)
			cudaMemcpy2D(data, data_pitch * sizeof(*data), ptr, sizeof(*ptr)*_pitch, _w * sizeof(*data), _h*_d, cudaMemcpyDeviceToDevice);
		else
			cudaMemcpy2DAsync(data, data_pitch * sizeof(*data), ptr, sizeof(*ptr)*_pitch, _w * sizeof(*data), _h*_d, cudaMemcpyDeviceToDevice, stream);
		CUDA_CHECK_ERROR;
	}
	inline bool BindToTexture1D(const textureReference* _tex, size_t offset = 0, cudaChannelFormatDesc* pDesc = NULL) const {
		bool bRtn = false;
		if (data) {
			cudaChannelFormatDesc desc = pDesc ? *pDesc : cudaCreateChannelDesc<T>();
			cudaBindTexture(&offset, _tex, data, &desc, width);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	inline bool BindToTexture2D(const textureReference* _tex, size_t offset = 0, cudaChannelFormatDesc* pDesc = NULL) const {
		bool bRtn = false;
		if (data) {
			cudaChannelFormatDesc desc = pDesc ? *pDesc : cudaCreateChannelDesc<T>();
			cudaBindTexture2D(&offset, _tex, data, &desc, width, height, data_pitch * sizeof(*data));
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	inline bool UnbindTexture(const textureReference* _tex) {
		cudaError err = cudaUnbindTexture(_tex);
		CUDA_CHECK_ERROR;
		return err == cudaSuccess ? true : false;
	}
	void Display(std::string name, size_t z = 0) const {
		if (z<0 || z >= depth) {
			assert(0);
			return;
		}
		const size_t size = width * height*depth;
		const size_t offset = z * width*height;
		std::vector<T> tmp(size + 1);
		T* buf = &tmp[0];
		this->CopyToHost(buf, width, width, height, depth);
		buf += offset;
	}
	T* data;
	size_t width, data_pitch, height, depth;
	std::vector<std::string> vecTexSymbol;
};


template<class T>
struct GpuData {
	typedef T DataType;
	typedef GpuDataRef<T> DataRef;
	inline virtual ~GpuData() {
		this->FreeBuffer();
	}
	inline GpuData() :width(0), height(0), depth(0), channels(0) {
		memset(&dev_data, 0, sizeof(dev_data));
		memset(&extent, 0, sizeof(extent));
	}
	inline GpuData(size_t w, size_t h = 1, size_t d = 1) : width(0), height(0), depth(0), channels(0) {
		memset(&dev_data, 0, sizeof(dev_data));
		memset(&extent, 0, sizeof(extent));
		MallocBuffer(w, h, d);
	}
	inline GpuData& operator=(const GpuData& obj) {
		MallocBuffer(obj.width, obj.height, obj.depth);
		GetDataRef().CopyFromDevice(obj.GetData(), obj.width, obj.DataPitch(), obj.height, obj.depth, stream.stream);
		return *this;
	}
	inline DataRef GetDataRef() const {
		return DataRef(GetData(), width, DataPitch(), height, depth);
	}
	inline GpuData& MallocBuffer(size_t w, size_t h, size_t d) {
		assert(h*d >= 1);
		if (width == w && height == h && depth == d) {
		}
		else {
			//reuse the device aligned buffer
			if (sizeof(T)*w < this->dev_data.pitch && h*d > 0 && h*d <= this->extent.height*this->extent.depth) {
				width = w;
				height = h;
				depth = d;
			}
			else {
				FreeBuffer();
				extent = make_cudaExtent(sizeof(T)*w, h, d);
				cudaMalloc3D(&dev_data, extent);
				CUDA_CHECK_ERROR;
				width = w;
				height = h;
				depth = d;
				if (d == 1) {
					if (h == 1) channels = 1;
					else        channels = 2;
				}
				else           channels = 3;
			}
		}
		return *this;
	}
	inline GpuData& MallocBuffer(size_t w, size_t h) {
		return MallocBuffer(w, h, 1);
	}
	inline GpuData& MallocBuffer(size_t w) {
		return MallocBuffer(w, 1, 1);
	}
	inline void FreeBuffer() {
		if (dev_data.ptr) {
			cudaFree(dev_data.ptr);
		}
		memset(&dev_data, 0, sizeof(dev_data));
		memset(&extent, 0, sizeof(extent));
		width = height = depth = 0;
		channels = 0;
	}
	inline bool BindToTexture1D(const textureReference* _tex, cudaChannelFormatDesc* pDesc = NULL) const {
		size_t offset = 0;
		return this->GetDataRef().BindToTexture1D(_tex, offset, pDesc);
	}
	inline bool BindToTexture2D(const textureReference* _tex, int nDepthIndex = 0, cudaChannelFormatDesc* pDesc = NULL) const {
		bool bRtn = false;
		size_t offset = this->DataPitch()*height*nDepthIndex;
		bRtn = this->GetDataRef().BindToTexture2D(_tex, offset, pDesc);
		return bRtn;
	}
	inline bool UnbindTexture(const textureReference* _tex) const {
		return this->GetDataRef().UnbindTexture(_tex);
	}
	inline void CopyToHost(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		GetDataRef().CopyToHost(ptr, _w, _pitch, _h, _d, stream.stream);
	}
	inline void CopyToDevice(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		GetDataRef().CopyToDevice(ptr, _w, _pitch, _h, _d, stream.stream);
	}
	inline void CopyFromHost(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		GetDataRef().CopyFromHost(ptr, _w, _pitch, _h, _d, stream.stream);
	}
	inline void CopyFromDevice(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		GetDataRef().CopyFromDevice(ptr, _w, _pitch, _h, _d, stream.stream);
	}
	inline GpuData& Zero() {
		GetDataRef().Zero(stream.stream);
		return *this;
	}
	inline GpuData& Ones() {
		GetDataRef().Ones(stream.stream);
		return *this;
	}
	inline GpuData& SetValue(T val) {
		GetDataRef().SetValue(val, stream.stream);
		return *this;
	}
	inline size_t DataPitch() const {
		return dev_data.pitch / sizeof(T);
	}
	inline T* GetData() const {
		return (T*)dev_data.ptr;
	}
	inline void Display(std::string name) const {
		GetDataRef().Display(name);
	}
	inline GpuData& SetStream(const GpuStream& s) {
		stream.stream = s.stream;
		return *this;
	}
	GpuStream stream;
	cudaPitchedPtr dev_data;
	cudaExtent extent;
	size_t width, height, depth, channels;
};

template<class T>
struct GpuBuffer {
	typedef T DataType;
	typedef GpuDataRef<T> DataRef;
	inline virtual ~GpuBuffer() {
		this->FreeBuffer();
	}
	inline GpuBuffer() :width(0), height(0), depth(0), channels(0), data(NULL) {
	}
	inline GpuBuffer(size_t w, size_t h = 1, size_t d = 1) : width(0), height(0), depth(0), channels(0), data(NULL) {
		MallocBuffer(w, h, d);
	}
	inline GpuBuffer& operator=(const GpuBuffer& obj) {
		MallocBuffer(obj.width, obj.height, obj.depth);
		GetDataRef().CopyFromDevice(obj.GetData(), obj.width, obj.DataPitch(), obj.height, obj.depth, stream.stream);
		return *this;
	}
	inline DataRef GetDataRef() const {
		return DataRef(GetData(), width, DataPitch(), height, depth);
	}
	bool Resize(size_t w, size_t h, size_t d) {
		bool bRtn = false;
		if (w*h*d <= total_size) {
			bRtn = true;
			width = w; height = h; depth = d;
		}
		return bRtn;
	}
	inline GpuBuffer& MallocBuffer(size_t w, size_t h, size_t d) {
		assert(h*d >= 1);
		if (width == w && height == h && depth == d) {
		}
		else {
			size_t _sz = width * height*depth;
			size_t sz = w * h*d;
			if (_sz >= sz) {
			}
			else {
				FreeBuffer();
				cudaMalloc(&data, sz * sizeof(T));
				_CUDA_CHECK_ERROR(CUDAString("w=%d, h=%d, d=%d", w, h, d));
			}
			width = w;
			height = h;
			depth = d;
			channels = 1;
		}
		total_size = width * height*depth;
		return *this;
	}
	inline GpuBuffer& MallocBuffer(size_t w, size_t h) {
		return MallocBuffer(w, h, 1);
	}
	inline GpuBuffer& MallocBuffer(size_t w) {
		return MallocBuffer(w, 1, 1);
	}
	inline void FreeBuffer() {
		if (data) cudaFree(data);
		data = NULL;
		width = height = depth = 0;
		channels = 0;
		total_size = 0;
	}
	inline bool BindToTexture1D(const textureReference* _tex, cudaChannelFormatDesc* pDesc = NULL) const;
	inline bool BindToTexture2D(const textureReference* _tex, int nDepthIndex = 0, cudaChannelFormatDesc* pDesc = NULL);
	inline bool UnbindTexture(const textureReference* _tex) const;
	inline void CopyToHost(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		assert(_w == _pitch);
		if (stream.stream == cudaStreamDefault)
			cudaMemcpy(ptr, data, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToHost);
		else
			cudaMemcpyAsync(ptr, data, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToHost, stream.stream);
		_CUDA_CHECK_ERROR(CUDAString("_w=%d, _pitch=%d, _h=%d, _d=%d", _w, _pitch, _h, _d));
	}
	inline void CopyToDevice(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		assert(_w == _pitch);
		if (stream.stream == cudaStreamDefault)
			cudaMemcpy(ptr, data, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToDevice);
		else
			cudaMemcpyAsync(ptr, data, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToDevice, stream.stream);
		_CUDA_CHECK_ERROR(CUDAString("_w=%d, _pitch=%d, _h=%d, _d=%d", _w, _pitch, _h, _d));
	}
	inline void CopyFromHost(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		assert(_w == _pitch);
		if (stream.stream == cudaStreamDefault)
			cudaMemcpy(data, ptr, sizeof(T)*_w*_h*_d, cudaMemcpyHostToDevice);
		else
			cudaMemcpyAsync(data, ptr, sizeof(T)*_w*_h*_d, cudaMemcpyHostToDevice, stream.stream);
		_CUDA_CHECK_ERROR(CUDAString("_w=%d, _pitch=%d, _h=%d, _d=%d", _w, _pitch, _h, _d));
	}
	inline void CopyFromDevice(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		assert(_w == _pitch);
		if (stream.stream == cudaStreamDefault)
			cudaMemcpy(data, ptr, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToDevice);
		else
			cudaMemcpyAsync(data, ptr, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToDevice, stream.stream);
		_CUDA_CHECK_ERROR(CUDAString("_w=%d, _pitch=%d, _h=%d, _d=%d", _w, _pitch, _h, _d));
	}
	inline GpuBuffer& Zero() {
		GetDataRef().Zero(stream.stream);
		return *this;
	}
	inline GpuBuffer& Ones() {
		GetDataRef().Ones(stream.stream);
		return *this;
	}
	inline GpuBuffer& SetValue(T val) {
		GetDataRef().SetValue(val, stream.stream);
		return *this;
	}
	inline size_t DataPitch() const {
		return width;
	}
	inline T* GetData() const {
		return (T*)data;
	}
	inline void Display(std::string name) const {
		GetDataRef().Display(name);
	}
	inline GpuBuffer& SetStream(const GpuStream& s) {
		stream.stream = s.stream;
		return *this;
	}
	GpuStream stream;
	T* data;
	size_t width, height, depth, channels;
	size_t total_size;
};

template<class T>
struct GpuArray2D {
	typedef T DataType;
	inline virtual ~GpuArray2D() {
		FreeBuffer();
	}
	inline GpuArray2D(unsigned int _flags = 0) :width(0), height(0), channels(2), dev_array(NULL), flags(_flags) {
	}
	inline GpuArray2D(size_t w, size_t h, unsigned int _flags = 0) : width(0), height(0), channels(2), dev_array(NULL), flags(_flags) {
		MallocBuffer(w, h);
	}
	inline GpuArray2D& operator=(const GpuArray2D& obj) {
		MallocBuffer(obj.width, obj.height, obj.flags);
		cudaMemcpy2DArrayToArray(this->dev_array, 0, 0, obj.dev_array, 0, 0, sizeof(T)*width, height, cudaMemcpyDeviceToDevice);
		CUDA_CHECK_ERROR;
		return *this;
	}
	//DataRef GetDataRef() const{
	//	return DataRef(GetData(), width, DataPitch(), height, depth); 
	//}
	inline GpuArray2D& MallocBuffer(size_t w, size_t h, unsigned int _flags = 0) {
		if (width == w && height == h) {
		}
		else {
			FreeBuffer();
			desc = cudaCreateChannelDesc<T>();
			cudaMallocArray(&dev_array, &desc, w, h, _flags);
			CUDA_CHECK_ERROR;
			width = w;
			height = h;
			flags = _flags;
			//std::cout<<"GpuArray2D_MallocBuffer:"<<uint(dev_array)<<std::endl;
		}
		return *this;
	}
	inline void FreeBuffer() {
		if (dev_array) {
			//std::cout<<"GpuArray2D_FreeBuffer:"<<uint(dev_array)<<std::endl;
			cudaFreeArray(dev_array);
			CUDA_CHECK_ERROR;
		}
		width = height = 0;
		flags = 0;
	}
	inline bool BindToTexture(const textureReference* _tex, cudaChannelFormatDesc* pDesc = NULL) const {
		bool bRtn = false;
		if (dev_array) {
			size_t offset = 0;
			::cudaBindTextureToArray(_tex, this->dev_array, pDesc ? pDesc : &desc);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	inline bool BindToSurface(const surfaceReference* _sur) {
		bool bRtn = false;
		if (flags != cudaArraySurfaceLoadStore) {
			std::cout << "flags != cudaArraySurfaceLoadStore" << std::endl;
			return false;
		}
		if (dev_array) {
			size_t offset = 0;
			::cudaBindSurfaceToArray(_sur, this->dev_array, &desc);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	inline bool UnbindTexture(const textureReference* _tex) {
		cudaError err = cudaUnbindTexture(_tex);
		CUDA_CHECK_ERROR;
		return err == cudaSuccess ? true : false;
	}
	template<class _Tp>
	inline GpuArray2D& SetValue(_Tp val) {
		int size = width * height;
		std::vector<T> buf(size + 1);
		T* pBuf = &buf[0];
		if (val != 0) {
			for (int i = 0; i<size; i++) pBuf[i] = val;
		}
		this->CopyFromHost(pBuf, width, width);
		//this->CopyFromDevice(devBuf.GetData(), devBuf.width, devBuf.DataPitch(), devBuf.height, devBuf.depth);
		return *this;
	}
	inline GpuArray2D& Zero() {
		this->SetValue(T(0));
		return *this;
	}
	inline GpuArray2D& Ones() {
		SetValue(T(1));
		return *this;
	}
	inline void CopyToHost(T* ptr, size_t _w, size_t _h) const {
		if (stream.stream == cudaStreamDefault) cudaMemcpyFromArray(ptr, dev_array, 0, 0, sizeof(T)*_w*_h, cudaMemcpyDeviceToHost);
		else                                    cudaMemcpyFromArrayAsync(ptr, dev_array, 0, 0, sizeof(T)*_w*_h, cudaMemcpyDeviceToHost, stream.stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyToDevice(T* ptr, size_t _w, size_t _h) const {
		if (stream.stream == cudaStreamDefault) cudaMemcpyFromArray(ptr, dev_array, 0, 0, sizeof(T)*_w*_h, cudaMemcpyDeviceToDevice);
		else                               cudaMemcpyFromArrayAsync(ptr, dev_array, 0, 0, sizeof(T)*_w*_h, cudaMemcpyDeviceToDevice, stream.stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyFromHost(const T* ptr, size_t _w, size_t _h) {
		if (stream.stream == cudaStreamDefault) cudaMemcpyToArray(dev_array, 0, 0, ptr, sizeof(T)*_w*_h, cudaMemcpyHostToDevice);
		else                               cudaMemcpyToArrayAsync(dev_array, 0, 0, ptr, sizeof(T)*_w*_h, cudaMemcpyHostToDevice, stream.stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyFromDevice(const T* ptr, size_t _w, size_t _h) {
		if (stream.stream == cudaStreamDefault) cudaMemcpyToArray(dev_array, 0, 0, ptr, sizeof(T)*_w*_h, cudaMemcpyDeviceToDevice);
		else                               cudaMemcpyToArrayAsync(dev_array, 0, 0, ptr, sizeof(T)*_w*_h, cudaMemcpyDeviceToDevice, stream.stream);
		CUDA_CHECK_ERROR;
	}
	inline void Display(std::string name = "") {
		size_t size = width * height;
		std::vector<T> buf(size + 1);
		T* pBuf = &buf[0];
		this->CopyToHost(pBuf, width, height);
	}
	inline GpuArray2D& SetStream(const GpuStream& s) {
		stream.stream = s.stream;
		return *this;
	}
	cudaArray*            dev_array;
	cudaChannelFormatDesc desc;
	size_t width, height;
	const size_t channels;
	unsigned int  flags;
	GpuStream             stream;
};

template<class T>
struct GpuArray3D {
	typedef T DataType;
	inline virtual ~GpuArray3D() {
		FreeBuffer();
	}
	inline GpuArray3D(unsigned int _flags = cudaArraySurfaceLoadStore) :width(0), height(0), depth(0), channels(3), dev_array(NULL), flags(_flags) {
		memset(&extent, 0, sizeof(extent));
	}
	inline GpuArray3D(size_t w, size_t h, size_t d, unsigned int _flags = cudaArraySurfaceLoadStore) : width(0), height(0), depth(0), channels(3), dev_array(NULL), flags(_flags) {
		memset(&extent, 0, sizeof(extent));
		MallocBuffer(w, h, d, _flags);
	}
	inline GpuArray3D& operator=(const GpuArray3D& obj) {
		MallocBuffer(obj.width, obj.height, obj.depth, obj.flags);
		cudaMemcpy3DParms param;
		param.srcArray = obj.dev_array;
		param.srcPos = make_cudaPos(0, 0, 0);
		param.dstArray = this->dev_array;
		param.dstPos = make_cudaPos(0, 0, 0);
		param.extent = this->extent;
		param.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&param);
		CUDA_CHECK_ERROR;
		return *this;
	}
	//DataRef GetDataRef() const{
	//	return DataRef(GetData(), width, DataPitch(), height, depth); 
	//}
	inline GpuArray3D& MallocBuffer(size_t w, size_t h, size_t d, unsigned int _flags = 0) {
		assert(h*d >= 1);
		if (width == w && height == h && depth == d) {
		}
		else {
			FreeBuffer();
			desc = cudaCreateChannelDesc<T>();
			extent = make_cudaExtent(w, h, d);
			cudaMalloc3DArray(&dev_array, &desc, extent, _flags);
			_CUDA_CHECK_ERROR(CUDAString("w=%d, h=%d, d=%d", w,  h, d));
			width = w;
			height = h;
			depth = d;
			flags = _flags;
		}
		return *this;
	}
	inline void FreeBuffer() {
		if (dev_array) {
			cudaFreeArray(dev_array);
		}
		width = height = depth = 0;
		flags = 0;
		memset(&extent, 0, sizeof(extent));
	}
	inline bool BindToTexture(const textureReference* _tex, cudaChannelFormatDesc* pDesc = NULL) const {
		bool bRtn = false;
		if (dev_array) {
			size_t offset = 0;
			::cudaBindTextureToArray(_tex, this->dev_array, pDesc ? pDesc : &desc);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	inline bool UnbindTexture(const textureReference* _tex) {
		cudaError err = cudaUnbindTexture(_tex);
		return err == cudaSuccess ? true : false;
	}
	inline bool BindToSurface(const surfaceReference* _sur) {
		bool bRtn = false;
		if (flags != cudaArraySurfaceLoadStore) {
			std::cout << "flags != cudaArraySurfaceLoadStore" << std::endl;
			return false;
		}
		if (dev_array) {
			size_t offset = 0;
			::cudaBindSurfaceToArray(_sur, this->dev_array, &desc);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	template<class _Tp>
	inline bool UnbindSurface() {
		cudaError err = cudaSuccess;
		assert(0);
		return err == cudaSuccess ? true : false;
	}
	template<class _Tp>
	inline GpuArray3D& SetValue(_Tp val) {
		int size = width * height*depth;
		std::vector<T> buf(size + 1);
		T* pBuf = &buf[0];
		if (val != 0) {
			for (int i = 0; i<size; i++) pBuf[i] = val;
		}
		this->CopyFromHost(pBuf, width, width, height, depth);
		return *this;
	}
	inline GpuArray3D& Zero() {
		this->SetValue(T(0));
		return *this;
	}
	inline GpuArray3D& Ones() {
		SetValue(T(1));
		return *this;
	}
	inline void CopyToHost(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d) const {
		cudaMemcpy3DParms param = { 0, };
		param.srcArray = this->dev_array;
		param.dstPtr = make_cudaPitchedPtr(ptr, _pitch * sizeof(ptr[0]), _w * sizeof(ptr[0]), _h);
		param.extent = this->extent;
		param.kind = cudaMemcpyDeviceToHost;
		if (stream.stream == cudaStreamDefault) cudaMemcpy3D(&param);
		else                               cudaMemcpy3DAsync(&param);
		_CUDA_CHECK_ERROR(CUDAString("_w=%d, _pitch=%d, _h=%d, _d=%d", _w, _pitch, _h, _d));
	}
	inline void CopyToDevice(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d) const {
		cudaMemcpy3DParms param = { 0, };
		param.srcArray = this->dev_array;
		param.dstPtr = make_cudaPitchedPtr(ptr, _pitch * sizeof(ptr[0]), _w * sizeof(ptr[0]), _h);
		param.extent = this->extent;
		param.kind = cudaMemcpyDeviceToDevice;
		if (stream.stream == cudaStreamDefault) cudaMemcpy3D(&param);
		else                               cudaMemcpy3DAsync(&param);
		_CUDA_CHECK_ERROR(CUDAString("_w=%d, _pitch=%d, _h=%d, _d=%d", _w, _pitch, _h, _d));
	}
	inline void CopyFromHost(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d) {
		cudaMemcpy3DParms param = { 0, };
		T* _ptr = ConstToGeneral(ptr);
		param.srcPtr = make_cudaPitchedPtr(_ptr, _pitch * sizeof(ptr[0]), _w * sizeof(ptr[0]), _h);
		param.dstArray = this->dev_array;
		param.extent = this->extent;
		param.kind = cudaMemcpyHostToDevice;
		if (stream.stream == cudaStreamDefault) cudaMemcpy3D(&param);
		else                               cudaMemcpy3DAsync(&param);
		_CUDA_CHECK_ERROR(CUDAString("_w=%d, _pitch=%d, _h=%d, _d=%d", _w, _pitch, _h, _d));
	}
	inline void CopyFromDevice(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d) {
		cudaMemcpy3DParms param = { 0, };
		T* _ptr = ConstToGeneral(ptr);
		param.srcPtr = make_cudaPitchedPtr(_ptr, _pitch * sizeof(ptr[0]), _w * sizeof(T), _h);
		param.dstArray = this->dev_array;
		param.extent = this->extent;
		param.kind = cudaMemcpyDeviceToDevice;
		if (stream.stream == cudaStreamDefault) cudaMemcpy3D(&param);
		else                               cudaMemcpy3DAsync(&param);
		_CUDA_CHECK_ERROR(CUDAString("_w=%d, _pitch=%d, _h=%d, _d=%d", _w, _pitch, _h, _d));
	}
	inline void Display(std::string name = "") {
		size_t size = width * height*depth;
		std::vector<T> buf(size + 1);
		T* pBuf = &buf[0];
		this->CopyToHost(pBuf, width, width, height, depth);
	}
	inline GpuArray3D& SetStream(const GpuStream& s) {
		stream.stream = s.stream;
		return *this;
	}
	cudaArray*            dev_array;
	cudaChannelFormatDesc desc;
	cudaExtent            extent;
	size_t width, height, depth;
	const size_t channels;
	unsigned int  flags;
	GpuStream             stream;
};

template<class T>
struct MappingData {
	typedef T DataType;
	inline MappingData() :width(0), height(0),
		depth(0), channels(0), hostData(NULL), GpuData(NULL),
		flags(cudaHostAllocMapped) {
	}
	inline MappingData(size_t w, size_t h = 1, size_t d = 1)
		: width(0), height(0), depth(0), channels(0), hostData(NULL), GpuData(NULL),
		flags(cudaHostAllocMapped) {
		MallocBuffer(w, h, d);
	}
	inline MappingData& MallocBuffer(size_t w, size_t h, size_t d) {
		if (width == w && height == h && depth == d) {
		}
		else {
			FreeBuffer();
			int size = sizeof(hostData[0])*w*h*d;
			if (size > 0) {
				cudaMallocHost(&hostData, size, flags);
				if (hostData) cudaHostGetDevicePointer(&GpuData, hostData, 0);
				CUDA_CHECK_ERROR;
				if (GpuData) {
					width = w; height = h; depth = d;
					if (d == 1) {
						if (h == 1) channels = 1;
						else channels = 2;
					}
					else channels = 3;
				}
			}
		}
		return *this;
	}
	inline void FreeBuffer() {
		if (hostData) {
			cudaFreeHost(hostData);
		}
		hostData = NULL;
		GpuData = NULL;
		width = height = depth = 0;
	}
	inline MappingData& SetValue(T val) {
		size_t size = width * height*depth;
		for (size_t i = 0; i<size; i++) hostData[i] = val;
		return *this;
	}
	inline MappingData& Zero() {
		memset(hostData, 0, sizeof(hostData[0])*width*height*depth);
		return *this;
	}
	inline MappingData& Ones() {
		return SetValue(T(1));
	}
	inline size_t DataPitch() const {
		return width;
	}
	inline T* GetHostData() const {
		return hostData;
	}
	inline T* GetGpuData() const {
		return GpuData;
	}
	inline void Display(std::string name) const {
		assert(0);
		//GetDataRef().Display(name);
	}
	const size_t flags;
	size_t width, height, depth;
	size_t channels;
	T* hostData;
	T* GpuData;
};


template<typename T> inline
cudaError_t LoadConstant(const void* cData, const T* data, int size) {
	cudaError_t err = cudaMemcpyToSymbol(cData, data, sizeof(data[0])*size);
	CUDA_CHECK_ERROR;
	return err;
}

template<typename T> inline
cudaError_t LoadConstant(const GpuStream& sm, const void* cData, const T* data, int size) {
	cudaError_t err = cudaSuccess;
	int offset = 0;
	if (sm.stream == cudaStreamDefault) err = cudaMemcpyToSymbol(cData, data, sizeof(data[0])*size, offset, cudaMemcpyHostToDevice);
	else err = cudaMemcpyToSymbolAsync(cData, data, sizeof(data[0])*size, offset, cudaMemcpyHostToDevice, sm.stream);
	CUDA_CHECK_ERROR;
	return err;
}

#if (CUDART_VERSION >= 9000)
#pragma message("CUDART_VERSION >= 9000")
#define __my_shfl_up(var, delta) __shfl_up_sync(0xFFFFFFFF, var, delta)
#define __my_shfl_down(var, delta) __shfl_down_sync(0xFFFFFFFF, var, delta)
#define __my_shfl(var, srcLane) __shfl_sync(0xFFFFFFFF, var, srcLane)
#else
#pragma message("CUDART_VERSION < 9000")
#define __my_shfl_up(var, delta) __shfl_up(var, delta)
#define __my_shfl_down(var, delta) __shfl_down(var, delta)
#define __my_shfl(var, srcLane) __shfl(var, srcLane)
#endif


struct gpuDevice {
	gpuDevice(int id = 0) {
		memset(&prop, 0, sizeof(prop));
		//cudaSetDevice(id);
		//CUDA_CHECK_ERROR;
		cudaGetDeviceProperties(&prop, id);
		CUDA_CHECK_ERROR;
	}
	cudaDeviceProp prop;
};

template<typename T>
struct ManagedBuffer {
	ManagedBuffer(long long _size) {
		buffer = cudaMallocManaged(&buffer, _size*sizeof(T));
		assert(buffer);
	}
	virtual ~ManagedBuffer() {
		if (buffer) cudaFree(&buffer);
	}
	T* buffer;
	long long  size;
};

inline size_t getFreeBytes(){
	size_t free_bytes;
	size_t total_bytes;
	cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
	CUDA_CHECK_ERROR;

	return free_bytes;
}

inline size_t getTotalBytes() {
	size_t free_bytes;
	size_t total_bytes;
	cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
	CUDA_CHECK_ERROR;

	return total_bytes;
}

#endif //__cudaLib_H