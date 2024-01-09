#ifndef _cudaBackProjection_H
#define _cudaBackProjection_H
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <assert.h>
#include <vector>
#include <string>

#include "../common/type.h"

#define STATIC_DEVICE



#define CUDA_CHECK_ERROR { \
	cudaError_t err = cudaGetLastError(); \
	if (cudaSuccess != err){ \
	std::cout<< "CUDA ERROR: " << cudaGetErrorString(err) << std::endl; \
	VERIFY_TRUE(0);\
	}\
}

#define CUFFT_CHECK_ERROR(result) \
    { \
    if (result != CUFFT_SUCCESS) \
	std::cout<< "CUFFT ERROR #" << result <<" "<<std::string(#result)<<std::endl; \
    }

__device__ __host__ __forceinline__
	int UpDivide(int x, int y){
		return (x+y-1)/y;
}

typedef struct __device_builtin__ __builtin_align__(16){
    int left, top, right, bottom;
}cuRECT;

static cuRECT cuMakeRECT(int x, int y, int w, int h){
	cuRECT rc;
	rc.left = x;
	rc.top = y;
	rc.right = x + w;
	rc.bottom = y + h;
	return rc;
}


template<class T> 
struct SharedMemory{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template<>
struct SharedMemory<double>{
	__device__ inline operator       double *(){
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
	__device__ inline operator const double *() const{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};

template<>
struct SharedMemory<float>{
	__device__ inline operator       float *(){
		extern __shared__ float __smem_f[];
		return (float *)__smem_f;
	}
	__device__ inline operator const float *() const{
		extern __shared__ float __smem_f[];
		return (float *)__smem_f;
	}
};

template<typename _Tp0, typename _Tp1> __device__ __forceinline__  
	bool IsInBox(_Tp0 i, _Tp0 j, _Tp1 _w, _Tp1 _h){
		return (i>=0 && i<=(_w-1) && j>=0 && j<=(_h-1))?true:false;
}

template<typename _Tp0, typename _Tp1> __device__ __forceinline__  
	bool IsInBox(_Tp0 i, _Tp0 j, _Tp0 k, _Tp1 _w, _Tp1 _h, _Tp1 _d){
		return (i>=0 && i<=(_w-1) && j>=0 && j<=(_h-1) && k>=0 && k<=(_d-1))?true:false;
}

template<typename T> __device__ __forceinline__  
	T& gpuPointValue(T* data, int nx, int ny, int nz, int x, int y, int z){
		assert(nx>0 && ny > 0 && nz > 0 && x>=0 && y>= 0 && z >= 0 && x<nx && y<ny && z < nz);
		return (data + nx*ny*z)[y*nx+x];
}

template<typename T> __device__ __forceinline__  
	T& gpuPointValue(T* data, int nx, int ny, int x, int y){
		return data[nx*y+x];
}

template<typename T> __device__ __forceinline__  
	T& gpuPointValue(T* data, int nx, int x){
		return data[x];
}

template<typename T>
cudaError_t LoadConstant(const void* cData, const T* data, int size){
	cudaError_t err = cudaMemcpyToSymbol(cData, data, sizeof(data[0])*size);
	CUDA_CHECK_ERROR;
	return err;
}

template<typename T>
struct DataRef{
	typedef T DataType;
	DataRef():data(0), width(0), height(0), depth(0), pitch(0), channels(0){
	}
	DataRef(T* _data, int _width, int _height, int _depth, int _pitch, int _channels)
		:data(_data), width(_width), height(_height), depth(_depth), pitch(_pitch), channels(_channels){
	}
	DataRef(const DataRef& obj){
		*this = obj;
	}
	DataRef& operator=(const DataRef& obj){
		data     = obj.data;
		width    = obj.width;
		height   = obj.height;
		depth    = obj.depth;
		pitch    = obj.pitch;
		channels = obj.channels;
		return *this;
	}
	bool Is3D() const { return channels==3;}
	bool Is2D() const { return channels==2 || (Is3D() && depth == 1) ;}
	bool Is1D() const { return channels==1;}
	DataRef& Make2D(){
		if (Is3D()){
			channels = 2;
			height *= depth;
			depth = 1;
		}
		return *this;
	}
	DataRef MakeDataRefFrom3D(int z = 0, int _depth = 0) const{
		int szXY = pitch*height;
		_depth = MAX(0, MIN(_depth, depth -z));
		T* buf = data + szXY*z;
		if (_depth == 0) buf = NULL;
		return DataRef(buf, width, height, _depth, pitch, channels);
	}
	DataRef MakeDataRefFrom2D(int y = 0, int _height = 0) const{
		int szXY = pitch*height;
		_height = MAX(0, MIN(_height, height - y));
		T* buf = data + pitch*y;
		if (_height == 0) buf = NULL;
		if (!Is2D()){
			buf = 0;
			_height = 0;
		}
		return DataRef(buf, width, _height, depth, pitch, channels);
	}

	void Display(std::string name, int z = 0){
		std::vector<T> tmp(width*height+1);
		T* pData = &tmp[0];
		DataRef refData = this->MakeDataRefFrom3D(z, 1);
		if (refData.data){
			cudaMemcpy2D(pData, sizeof(*pData)*width, refData.data, refData.pitch*sizeof(*pData), sizeof(*pData)*width, refData.height, cudaMemcpyDeviceToHost);
			CUDA_CHECK_ERROR;
			::ShowImage(name, pData, width, height);
		}else{
			assert(0);
		}
	}

	template<typename _Tp>
	void Display(std::string name, int z, _Tp func){
		std::vector<T> tmp(width*height+1);
		T* pData = &tmp[0];
		DataRef refData = this->MakeDataRefFrom3D(z, 1);
		if (refData.data){
			cudaMemcpy2D(pData, sizeof(*pData)*width, refData.data, refData.pitch*sizeof(*pData), sizeof(*pData)*width, refData.height, cudaMemcpyDeviceToHost);
			CUDA_CHECK_ERROR;
			std::vector<float> val(width*height+1);
			std::transform(tmp.begin(), tmp.end(), val.begin(), func);
			::ShowImage(name, &val[0], width, height);
		}else{
			assert(0);
		}
	}

	bool MemcpyDeviceToHost(T* pData, int _width, int _height = 1, int _depth = 1){
		if (_width*_height*_depth > width*height*depth) return false;
		const int HD = height*depth;
		if (HD > 1) cudaMemcpy2D(pData, sizeof(pData[0])*_width, data, pitch*sizeof(data[0]), sizeof(pData[0])*_width, HD, cudaMemcpyDeviceToHost);
		else        cudaMemcpy(pData, data, sizeof(pData[0])*_width, cudaMemcpyDeviceToHost);	
		CUDA_CHECK_ERROR;
		return true;
	}

	bool MemcpyHostToDevice(const T* pData, int _width, int _height = 1, int _depth = 1){
		if (_width*_height*_depth > width*height*depth) return false;
		const int HD = _height*_depth;
		if (HD > 1) cudaMemcpy2D(data, pitch*sizeof(data[0]), pData, sizeof(pData[0])*_width, sizeof(pData[0])*_width, HD, cudaMemcpyHostToDevice);
		else        cudaMemcpy(pData,  data, sizeof(pData[0])*_width, cudaMemcpyHostToDevice);	
		CUDA_CHECK_ERROR;
		return true;
	}

	bool MemcpyToDevice(T* pData, int _width, int _pitch, int _height = 1, int _depth = 1){
		if (_width*_height*_depth > width*height*depth) return false;
		const int HD = height*depth;
		if (HD > 1) cudaMemcpy2D(pData, sizeof(pData[0])*_pitch, data, pitch*sizeof(data[0]), sizeof(pData[0])*_width, HD, cudaMemcpyDeviceToDevice);
		else        cudaMemcpy(pData, data, sizeof(pData[0])*_width, cudaMemcpyDeviceToDevice);	
		CUDA_CHECK_ERROR;
		return true;
	}

	int width, height, depth, pitch, channels;
	T* data;
};

template<typename T, int CHANNEL_DIM>
struct gpuData{
	typedef T DataType;
	typedef DataRef<T> tDataRef;
	gpuData():channels(CHANNEL_DIM),width(0),height(0),pitch(0),dev_data(NULL){
	}
	gpuData(int w, int h):channels(CHANNEL_DIM),width(0),height(0),pitch(0),dev_data(NULL){
		MallocBuffer(w, h);
	}
	virtual ~gpuData(){
		FreeBuffer();
	}
	DataRef<T> GetDataRef() const{
		return DataRef<T>(Data(), width, height, 1, DataPitch());
	}
	T* Data() const{
		return (T*)this->dev_data;
	}
	size_t DataPitch() const{
		return pitch/sizeof(T);
	}
	bool MallocBuffer(int w, int h){
		bool bRtn = false;
		if (w == width && height == h){
			bRtn = true;
		}else{
			FreeBuffer();
			cudaMallocPitch(&dev_data, &pitch, sizeof(T)*w, h);
			CUDA_CHECK_ERROR;
			width = w;
			height = h;
			bRtn = true;
		}
		return bRtn;
	}
	void FreeBuffer(){
		if (dev_data){
			::cudaFree(dev_data);
			CUDA_CHECK_ERROR;
			dev_data = NULL;
			width = height = pitch = 0;
		}
	}
	bool CopyHostToDevice(const T* data, int w, int h){
		bool bRtn = false;
		if (MallocBuffer(w, h)){
			::cudaMemcpy2D(dev_data, pitch, data, sizeof(*data)*w, sizeof(*data)*w, h, cudaMemcpyHostToDevice);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	bool CopyDeviceToHost(T* data, int w, int h){
		bool bRtn = false;
		assert(w == width && h == height);
		if (data && dev_data){
			::cudaMemcpy2D(data, sizeof(*data)*w, dev_data, pitch, sizeof(*dev_data)*w, h, cudaMemcpyDeviceToHost);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	template<class _Tp>
	bool BindToTexture(_Tp& _tex);
	gpuData& SetValue(T value){
		int size = width*height;
		std::vector<T> tmp(size+1);
		T* pData = &tmp[0];
		for (int i=0; i<size; i ++) pData[i] = value;
		CopyHostToDevice(pData, width, height);
		return *this;
	}

	gpuData& Zero(){
		cudaMemset(dev_data, 0, pitch*height);
		CUDA_CHECK_ERROR;
		return *this;
	}

	template<class _Tp>
	void Display(std::string name, _Tp& func){
		std::vector<T> src(width*height+1);
		bool bRtn = CopyDeviceToHost(&src[0], width, height);
		if (bRtn){
			std::vector<float> dst(width*height+1);
			for (int i=0; i<width*height; i ++){
				dst[i] = func(src[i]);
			}
			ShowImage(name, &dst[0], width, height);
		}
	}
public:
	T* dev_data;
	size_t width, height;
	size_t pitch;
	const int channels; 
};

template<>
struct gpuData<float, 1>{
	typedef float DataType;
	typedef DataRef<float> DataRefType;
	gpuData():channels(1),width(0),pitch(0),dev_data(NULL){
	}
	gpuData(int w):channels(1),width(0),pitch(0),dev_data(NULL){
		MallocBuffer(w);
	}
	virtual ~gpuData(){
		FreeBuffer();
	}
	DataRef<float> GetDataRef() const{
		return DataRef<float>(Data(), width, 1, 1, DataPitch(), channels);
	}
	float* Data() const{
		return (float*)this->dev_data;
	}
	size_t DataPitch() const{
		return pitch/sizeof(float);
	}
	bool MallocBuffer(int w){
		bool bRtn = false;
		if (w == width){
			bRtn = true;
		}else{
			FreeBuffer();
			cudaMallocPitch(&dev_data, &pitch, sizeof(float)*w, 1);
			CUDA_CHECK_ERROR;
			width = w;
			bRtn = true;
		}
		return bRtn;
	}
	void FreeBuffer(){
		if (dev_data){
			::cudaFree(dev_data);
			CUDA_CHECK_ERROR;
			dev_data = NULL;
			width = pitch = 0;
		}
	}
	bool CopyHostToDevice(const float* data, int w){
		bool bRtn = false;
		if (MallocBuffer(w)){
			//::cudaMemcpy2D(dev_data, pitch, data, sizeof(*data)*w, sizeof(*data)*w, h, cudaMemcpyHostToDevice);
			::cudaMemcpy(dev_data, data, sizeof(*data)*w, cudaMemcpyHostToDevice);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	bool CopyDeviceToHost(float* data, int w){
		bool bRtn = false;
		assert(w == width);
		if (data && dev_data){
			//::cudaMemcpy2D(data, sizeof(*data)*w, dev_data, pitch, sizeof(*dev_data)*w, h, cudaMemcpyDeviceToHost);
			::cudaMemcpy(data, dev_data, sizeof(*data)*w, cudaMemcpyDeviceToHost);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	template<class _Tp>
	bool BindToTexture(_Tp& _tex){
		bool bRtn = false;
		if (dev_data){
			size_t offset = 0;
			cudaChannelFormatDesc floatChannel = cudaCreateChannelDesc<float> ();
			cudaBindTexture2D(&offset, _tex, dev_data, floatChannel, width, 1, pitch);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	gpuData& SetValue(float value){
		int size = width;
		std::vector<float> tmp(size+1);
		float* pData = &tmp[0];
		for (int i=0; i<size; i ++) pData[i] = value;
		CopyHostToDevice(pData, width);
		return *this;
	}

	gpuData& Zero(){
		cudaMemset(dev_data, 0, pitch);
		CUDA_CHECK_ERROR;
		return *this;
	}
	void Display(std::string name){
		const int HEIGHT = 128;
		std::vector<float> tmp(width*HEIGHT+1);
		float* pData = &tmp[0];
		this->CopyDeviceToHost(pData, width);
		for (int i=1; i<HEIGHT; i ++){
			memcpy(pData + width*i, pData, sizeof(pData[0])*width);
		}
		ShowImage(name, pData, width, HEIGHT);
	}
public:
	float* dev_data;
	size_t width;
	size_t pitch;
	const int channels; 
};

template<>
struct gpuData<float, 2>{
	typedef float DataType;
	typedef DataRef<float> DataRefType;
	gpuData():channels(2),width(0),height(0),pitch(0),dev_data(NULL){
	}
	gpuData(int w, int h):channels(2),width(0),height(0),pitch(0),dev_data(NULL){
		MallocBuffer(w, h);
	}
	virtual ~gpuData(){
		FreeBuffer();
	}
	gpuData& operator=(const gpuData& obj){
		this->MallocBuffer(obj.width, obj.height);
		obj.GetDataRef().MemcpyToDevice(this->Data(), width, this->DataPitch(), height, 1);
		return *this;
	}
	DataRef<float> GetDataRef() const{
		return DataRef<float>(Data(), width, height, 1, DataPitch(), channels);
	}
	float* Data() const{
		return (float*)this->dev_data;
	}
	size_t DataPitch() const{
		return pitch/sizeof(float);
	}
	bool MallocBuffer(int w, int h){
		bool bRtn = false;
		if (w == width && height == h){
			bRtn = true;
		}else{
			FreeBuffer();
			cudaMallocPitch(&dev_data, &pitch, sizeof(float)*w, h);
			CUDA_CHECK_ERROR;
			width = w;
			height = h;
			bRtn = true;
		}
		return bRtn;
	}
	void FreeBuffer(){
		if (dev_data){
			::cudaFree(dev_data);
			CUDA_CHECK_ERROR;
			dev_data = NULL;
			width = height = pitch = 0;
		}
	}
	bool CopyHostToDevice(const float* data, int w, int h){
		bool bRtn = false;
		assert(w == width && h == height);
		if (MallocBuffer(w, h)){
			::cudaMemcpy2D(dev_data, pitch, data, sizeof(*data)*w, sizeof(*data)*w, h, cudaMemcpyHostToDevice);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	bool CopyDeviceToHost(float* data, int w, int h){
		assert(w == width && h == height);
		bool bRtn = false;
		assert(w == width && h == height);
		if (data && dev_data){
			::cudaMemcpy2D(data, sizeof(*data)*w, dev_data, pitch, sizeof(*dev_data)*w, h, cudaMemcpyDeviceToHost);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	template<class _Tp>
	bool BindToTexture(_Tp& _tex){
		bool bRtn = false;
		if (dev_data){
			size_t offset = 0;
			cudaChannelFormatDesc floatChannel = cudaCreateChannelDesc<float> ();
			cudaBindTexture2D(&offset, _tex, dev_data, floatChannel, width, height, pitch);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	gpuData& SetValue(float value){
		int size = width*height;
		std::vector<float> tmp(size+1);
		float* pData = &tmp[0];
		for (int i=0; i<size; i ++) pData[i] = value;
		CopyHostToDevice(pData, width, height);
		return *this;
	}

	gpuData& Zero(){
		cudaMemset(dev_data, 0, pitch*height);
		CUDA_CHECK_ERROR;
		return *this;
	}

	void Display(std::string name){
		this->GetDataRef().Display(name, 0);
	}

public:
	float* dev_data;
	size_t width, height;
	size_t pitch;
	const int channels; 

};

template<>
struct gpuData<double, 2>{
	typedef double DataType;
	typedef DataRef<double> DataRefType;
	gpuData():channels(2),width(0),height(0),pitch(0),dev_data(NULL){
	}
	gpuData(int w, int h):channels(2),width(0),height(0),pitch(0),dev_data(NULL){
		MallocBuffer(w, h);
	}
	virtual ~gpuData(){
		FreeBuffer();
	}
	DataRef<double> GetDataRef() const{
		return DataRef<double>(Data(), width, height, 1, DataPitch(), channels);
	}
	double* Data() const{
		return (double*)this->dev_data;
	}
	size_t DataPitch() const{
		return pitch/sizeof(double);
	}
	bool MallocBuffer(int w, int h){
		bool bRtn = false;
		if (w == width && height == h){
			bRtn = true;
		}else{
			FreeBuffer();
			cudaMallocPitch(&dev_data, &pitch, sizeof(dev_data[0])*w, h);
			CUDA_CHECK_ERROR;
			width = w;
			height = h;
			bRtn = true;
		}
		return bRtn;
	}
	void FreeBuffer(){
		if (dev_data){
			::cudaFree(dev_data);
			CUDA_CHECK_ERROR;
			dev_data = NULL;
			width = height = pitch = 0;
		}
	}
	bool CopyHostToDevice(const double* data, int w, int h){
		bool bRtn = false;
		assert(w == width && h == height);
		if (MallocBuffer(w, h)){
			::cudaMemcpy2D(dev_data, pitch, data, sizeof(*data)*w, sizeof(*data)*w, h, cudaMemcpyHostToDevice);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	bool CopyDeviceToHost(double* data, int w, int h){
		assert(w == width && h == height);
		bool bRtn = false;
		assert(w == width && h == height);
		if (data && dev_data){
			::cudaMemcpy2D(data, sizeof(*data)*w, dev_data, pitch, sizeof(*dev_data)*w, h, cudaMemcpyDeviceToHost);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	template<class _Tp>
	bool BindToTexture(_Tp& _tex){
		bool bRtn = false;
		if (dev_data){
			size_t offset = 0;
			cudaChannelFormatDesc doubleChannel = cudaCreateChannelDesc<double> ();
			cudaBindTexture2D(&offset, _tex, dev_data, doubleChannel, width, height, pitch);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	gpuData& SetValue(double value){
		int size = width*height;
		std::vector<double> tmp(size+1);
		double* pData = &tmp[0];
		for (int i=0; i<size; i ++) pData[i] = value;
		CopyHostToDevice(pData, width, height);
		return *this;
	}

	gpuData& Zero(){
		cudaMemset(dev_data, 0, pitch*height);
		CUDA_CHECK_ERROR;
		return *this;
	}

	void Display(std::string name){
		this->GetDataRef().Display(name, 0);
	}

public:
	double* dev_data;
	size_t width, height;
	size_t pitch;
	const int channels; 

};

template<>
struct gpuData<float, 3>{
	typedef float DataType;
	typedef DataRef<float> DataRefType;
	gpuData():channels(3), width(0), height(0), depth(0){
		memset(&extent, 0, sizeof(extent));
		memset(&dev_data, 0, sizeof(dev_data));
	}
	gpuData(int w, int h, int d):channels(3), width(0), height(0), depth(0){
		memset(&extent, 0, sizeof(extent));
		memset(&dev_data, 0, sizeof(dev_data));
		MallocBuffer(w, h, d);
	}
	gpuData& operator=(const gpuData& _data){
		MallocBuffer(_data.width, _data.height, _data.depth);
		cudaMemcpy(dev_data.ptr, _data.dev_data.ptr, dev_data.pitch*height*depth, cudaMemcpyDeviceToDevice);
		CUDA_CHECK_ERROR;
		return *this;
	}
	virtual ~gpuData(){
		FreeBuffer();
	}
	DataRef<float> GetDataRef() const{
		return DataRef<float>(Data(), width, height, depth, DataPitch(), channels);
	}
	DataRef<float> GetDataRef2D() const{
		return GetDataRef().Make2D();
	}
	float* Data() const{
		return (float*)this->dev_data.ptr;
	}
	size_t DataPitch() const{
		return dev_data.pitch/sizeof(float);
	}
	gpuData& Zero(){
		if (this->dev_data.ptr){
			//cudaMemset(dev_data.ptr, 0, dev_data.pitch*height*depth);
			cudaMemset3D(dev_data, 0, extent);
			CUDA_CHECK_ERROR;
		}
		return *this;
	}
	gpuData& Ones(){
		return SetValue(1.);
	}
	gpuData& SetValue(float value){
		if (this->dev_data.ptr){
			int size = width*height*depth;
			std::vector<float> tmp(size + 1);
			float* data = &tmp[0];
			if (data){
				for (int i=0; i<size; i ++) data[i] = value;
				CopyHostToDevice(data, width, height, depth);
			}
		}
		return *this;
	}
	bool MallocBuffer(int w, int h, int d){
		bool bRtn = false;
		if (w == width && height == h && depth == d){
			bRtn = true;
		}else{
			FreeBuffer();
			extent =  make_cudaExtent(sizeof(float)*w, h, d);
			//cudaMallocPitch(&dev_data, &pitch, sizeof(float)*w, h);
			cudaMalloc3D(&dev_data, extent);
			CUDA_CHECK_ERROR;
			width = w; height = h; depth = d;
			bRtn = true;
		}
		return bRtn;
	}
	void FreeBuffer(){
		if (dev_data.ptr){
			::cudaFree(dev_data.ptr);
			CUDA_CHECK_ERROR;
			dev_data.ptr = NULL;
			width = height = depth = 0;
		}
	}
	bool CopyHostToDevice(float* data, int w, int h, int d){
		bool bRtn = false;
		if (MallocBuffer(w, h, d) && data){
			if (dev_data.pitch == dev_data.xsize){
				cudaMemcpy(dev_data.ptr, data, w*h*d*sizeof(float), cudaMemcpyHostToDevice);
				CUDA_CHECK_ERROR;
			}else{
				char* src = NULL;
				char* dst = NULL;
				for (int i= 0; i<d; i ++){
					for (int j=0; j<h; j ++){
						src = (char*)(data + i*w*h+ j*w);
						dst = (char*)dev_data.ptr + i*dev_data.pitch*dev_data.ysize + j*dev_data.pitch;
						cudaMemcpyAsync(dst,  src, w*sizeof(float), cudaMemcpyHostToDevice);
						CUDA_CHECK_ERROR;
					}
				}
				//sync the device
				cudaMemcpy(dst,  src, w*sizeof(float), cudaMemcpyHostToDevice);
				CUDA_CHECK_ERROR;
			}
			width = w;
			height = h;
			depth = d;
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	bool CopyDeviceToHost(float* data, int w, int h, int d){
		bool bRtn = false;
		assert(w == width && h == height && depth == d);
		if (data && dev_data.ptr){
			if (dev_data.pitch == dev_data.xsize){
				cudaMemcpy(data, dev_data.ptr, dev_data.pitch*h*d, cudaMemcpyDeviceToHost);
				CUDA_CHECK_ERROR;
			}else{
				char* src = NULL;
				char* dst = NULL;
				const int szXYdst = w*h;
				const int szXYsrc = dev_data.pitch*dev_data.ysize;
				for (int i= 0; i<d; i ++){
					for (int j=0; j<h; j ++){
						dst = (char*)(data + i*szXYdst+ j*w);
						src = (char*)dev_data.ptr + i*szXYsrc + j*dev_data.pitch;
						cudaMemcpyAsync(dst,  src, w*sizeof(float), cudaMemcpyDeviceToHost);
					}
				}
				//sync the device
				cudaMemcpy(dst,  src, w*sizeof(data[0]), cudaMemcpyDeviceToHost);
				CUDA_CHECK_ERROR;
			}
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	bool CopyToDevice(float* data, int _width, int _pitch, int _height, int _depth){
		assert(_width == width && _height == height && _depth == depth);
		return this->GetDataRef().MemcpyToDevice(data, _width, _pitch, _height, _depth);
	}
	float* GetDataZ(int z) const{
		if (z>=0 && z<=depth-1)	return (float*)((char*)dev_data.ptr + dev_data.pitch*dev_data.ysize*z);
		else return NULL;
	}
	bool GetDataZ(float* pData, int w, int h, int z) const{
		bool bRtn = false;
		const float* pDataSrc = GetDataZ(z);
		if (pDataSrc && width == w && height == h){
			cudaMemcpy2D(pData, w*sizeof(*pData), pDataSrc, dev_data.pitch, sizeof(*pDataSrc)*width, height, cudaMemcpyDeviceToHost);
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	template<class _Tp>
	bool BindToTexture(_Tp& _tex, int z) const{
		bool bRtn = false;
		if (z >= 0 && z <= depth-1){
#ifdef _WIN32
			size_t offset = 0;
			cudaChannelFormatDesc floatChannel = cudaCreateChannelDesc<float> ();
			cudaBindTexture2D(&offset, _tex, GetDataZ(z), floatChannel, width, height, pitch);
			CUDA_CHECK_ERROR;
#else
			assert(0);
#endif
			bRtn = true;
		}
		assert(0);
		return bRtn;
	}
	void Display(std::string name, int z) const{
		this->GetDataRef().Display(name, z);
	}
	cudaPitchedPtr dev_data;
	cudaExtent extent;
	int width, height, depth;
	const int channels; 

};

template<>
struct gpuData<float2, 3>{
	typedef float2 DataType;
	typedef DataRef<float2> DataRefType;
	gpuData():channels(3), width(0), height(0), depth(0){
		memset(&extent, 0, sizeof(extent));
		memset(&dev_data, 0, sizeof(dev_data));
	}
	gpuData(int w, int h, int d):channels(3), width(0), height(0), depth(0){
		memset(&extent, 0, sizeof(extent));
		memset(&dev_data, 0, sizeof(dev_data));
		MallocBuffer(w, h, d);
	}
	gpuData& operator=(const gpuData& _data){
		MallocBuffer(_data.width, _data.height, _data.depth);
		cudaMemcpy(dev_data.ptr, _data.dev_data.ptr, dev_data.pitch*height*depth, cudaMemcpyDeviceToDevice);
		CUDA_CHECK_ERROR;
		return *this;
	}
	virtual ~gpuData(){
		FreeBuffer();
	}
	DataRef<float2> GetDataRef() const{
		return DataRef<float2>(Data(), width, height, depth, DataPitch(), channels);
	}
	DataRef<float2> GetDataRef2D() const{
		return GetDataRef().Make2D();
	}
	float2* Data() const{
		return (float2*)this->dev_data.ptr;
	}
	size_t DataPitch() const{
		return dev_data.pitch/sizeof(float2);
	}
	gpuData& Zero(){
		if (this->dev_data.ptr){
			//cudaMemset(dev_data.ptr, 0, dev_data.pitch*height*depth);
			cudaMemset3D(dev_data, 0, extent);
			CUDA_CHECK_ERROR;
		}
		return *this;
	}
	gpuData& Ones(){
		return SetValue(make_float2(1, 0));
	}
	gpuData& SetValue(float2 value){
		if (this->dev_data.ptr){
			int size = width*height*depth;
			std::vector<float2> tmp(size + 1);
			float2* data = &tmp[0];
			if (data){
				for (int i=0; i<size; i ++) data[i] = value;
				CopyHostToDevice(data, width, height, depth);
			}
		}
		return *this;
	}
	bool MallocBuffer(int w, int h, int d){
		bool bRtn = false;
		if (w == width && height == h && depth == d){
			bRtn = true;
		}else{
			FreeBuffer();
			extent =  make_cudaExtent(sizeof(float2)*w, h, d);
			//cudaMallocPitch(&dev_data, &pitch, sizeof(float2)*w, h);
			cudaMalloc3D(&dev_data, extent);
			CUDA_CHECK_ERROR;
			width = w; height = h; depth = d;
			bRtn = true;
		}
		return bRtn;
	}
	void FreeBuffer(){
		if (dev_data.ptr){
			::cudaFree(dev_data.ptr);
			CUDA_CHECK_ERROR;
			dev_data.ptr = NULL;
			width = height = depth = 0;
		}
	}
	bool CopyHostToDevice(float2* data, int w, int h, int d){
		bool bRtn = false;
		if (MallocBuffer(w, h, d) && data){
			if (dev_data.pitch == dev_data.xsize){
				cudaMemcpy(dev_data.ptr, data, w*h*d*sizeof(float2), cudaMemcpyHostToDevice);
			}else{
				char* src = NULL;
				char* dst = NULL;
				for (int i= 0; i<d; i ++){
					for (int j=0; j<h; j ++){
						src = (char*)(data + i*w*h+ j*w);
						dst = (char*)dev_data.ptr + i*dev_data.pitch*dev_data.ysize + j*dev_data.pitch;
						cudaMemcpyAsync(dst,  src, w*sizeof(float2), cudaMemcpyHostToDevice);
					}
				}
				//sync the device
				cudaMemcpy(dst,  src, w*sizeof(float2), cudaMemcpyHostToDevice);
			}
			width = w;
			height = h;
			depth = d;
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	bool CopyDeviceToHost(float2* data, int w, int h, int d){
		bool bRtn = false;
		assert(w == width && h == height && depth == d);
		if (data && dev_data.ptr){
			if (dev_data.pitch == dev_data.xsize){
				cudaMemcpy(data, dev_data.ptr, dev_data.pitch*h*d, cudaMemcpyDeviceToHost);
			}else{
				char* src = NULL;
				char* dst = NULL;
				const int szXYdst = w*h;
				const int szXYsrc = dev_data.pitch*dev_data.ysize;
				for (int i= 0; i<d; i ++){
					for (int j=0; j<h; j ++){
						dst = (char*)(data + i*szXYdst+ j*w);
						src = (char*)dev_data.ptr + i*szXYsrc + j*dev_data.pitch;
						cudaMemcpyAsync(dst,  src, w*sizeof(float2), cudaMemcpyDeviceToHost);
					}
				}
				//sync the device
				cudaMemcpy(dst,  src, w*sizeof(data[0]), cudaMemcpyDeviceToHost);
			}
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	float2* GetDataZ(int z) const{
		if (z>=0 && z<=depth-1)	return (float2*)((char*)dev_data.ptr + dev_data.pitch*dev_data.ysize*z);
		else return NULL;
	}
	bool GetDataZ(float2* pData, int w, int h, int z){
		bool bRtn = false;
		const float2* pDataSrc = GetDataZ(z);
		if (pDataSrc && width == w && height == h){
			cudaMemcpy2D(pData, w*sizeof(*pData), pDataSrc, sizeof(*pDataSrc)*width, sizeof(*pDataSrc)*width, height, cudaMemcpyHostToDevice);
			CUDA_CHECK_ERROR;
		}
		return bRtn;
	}
	template<class _Tp>
	bool BindToTexture(_Tp& _tex, int z) const{
		bool bRtn = false;
		if (z >= 0 && z <= depth-1){
			size_t offset = 0;
			cudaChannelFormatDesc float2Channel = cudaCreateChannelDesc<float2> ();
#ifdef _WIN32
			cudaBindTexture2D(&offset, _tex, GetDataZ(z), float2Channel, width, height, pitch);
#else
			assert(0);
#endif
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		assert(0);
		return bRtn;
	}
	template<class _Tp>
	void Display(std::string name, int z, _Tp func) const{
		this->GetDataRef().Display(name, z, func);
		//std::vector<float2> tmp(width*height+1);
		//float2* pDataSrc = GetDataZ(z);
		//float2* pDataDst = &tmp[0];
		//if (pDataSrc){
		//	::cudaMemcpy2D(pDataDst, width*sizeof(*pDataDst), pDataSrc, sizeof(*pDataSrc)*width, sizeof(*pDataSrc)*width, height, cudaMemcpyHostToDevice);
		//	CUDA_CHECK_ERROR;
		//	std::vector<float> val(width*height+1);
		//	for (int i=0; i<val.size(); i ++) val[i] = func(tmp[i]);
		//	ShowImage(name.c_str(), &val[0], width, height);
		//}
	}
	cudaPitchedPtr dev_data;
	cudaExtent extent;
	int width, height, depth;
	const int channels; 

};

typedef unsigned short ushort;

template<>
struct gpuData<ushort, 3>{
	typedef ushort DataType;
	typedef DataRef<ushort> DataRefType;
	gpuData():channels(3), width(0), height(0), depth(0){
		memset(&extent, 0, sizeof(extent));
		memset(&dev_data, 0, sizeof(dev_data));
	}
	gpuData(int w, int h, int d):channels(3), width(0), height(0), depth(0){
		memset(&extent, 0, sizeof(extent));
		memset(&dev_data, 0, sizeof(dev_data));
		MallocBuffer(w, h, d);
	}
	gpuData& operator=(const gpuData& _data){
		MallocBuffer(_data.width, _data.height, _data.depth);
		cudaMemcpy(dev_data.ptr, _data.dev_data.ptr, dev_data.pitch*height*depth, cudaMemcpyDeviceToDevice);
		CUDA_CHECK_ERROR;
		return *this;
	}
	virtual ~gpuData(){
		FreeBuffer();
	}
	DataRef<ushort> GetDataRef() const{
		return DataRef<ushort>(Data(), width, height, depth, DataPitch(), channels);
	}
	DataRef<ushort> GetDataRef2D() const{
		return GetDataRef().Make2D();
	}
	ushort* Data() const{
		return (ushort*)this->dev_data.ptr;
	}
	size_t DataPitch() const{
		return dev_data.pitch/sizeof(ushort);
	}
	gpuData& Zero(){
		if (this->dev_data.ptr){
			//cudaMemset(dev_data.ptr, 0, dev_data.pitch*height*depth);
			cudaMemset3D(dev_data, 0, extent);
			CUDA_CHECK_ERROR;
		}
		return *this;
	}
	gpuData& Ones(){
		return SetValue(1.);
	}
	gpuData& SetValue(ushort value){
		if (this->dev_data.ptr){
			int size = width*height*depth;
			std::vector<ushort> tmp(size + 1);
			ushort* data = &tmp[0];
			if (data){
				for (int i=0; i<size; i ++) data[i] = value;
				CopyHostToDevice(data, width, height, depth);
				CUDA_CHECK_ERROR;
			}
		}
		return *this;
	}
	bool MallocBuffer(int w, int h, int d){
		bool bRtn = false;
		if (w == width && height == h && depth == d){
			bRtn = true;
		}else{
			FreeBuffer();
			extent =  make_cudaExtent(sizeof(ushort)*w, h, d);
			//cudaMallocPitch(&dev_data, &pitch, sizeof(ushort)*w, h);
			cudaMalloc3D(&dev_data, extent);
			CUDA_CHECK_ERROR;
			width = w; height = h; depth = d;
			bRtn = true;
		}
		return bRtn;
	}
	void FreeBuffer(){
		if (dev_data.ptr){
			::cudaFree(dev_data.ptr);
			CUDA_CHECK_ERROR;
			dev_data.ptr = NULL;
			width = height = depth = 0;
		}
	}
	bool CopyHostToDevice(ushort* data, int w, int h, int d){
		assert(w == width && h == height && d == depth);
		bool bRtn = false;
		if (MallocBuffer(w, h, d) && data){
			if (dev_data.pitch == sizeof(ushort)*w){
				cudaMemcpy(dev_data.ptr, data, w*h*d*sizeof(ushort), cudaMemcpyHostToDevice);
			}else{
				char* src = NULL;
				char* dst = NULL;
				for (int i= 0; i<d; i ++){
					for (int j=0; j<h; j ++){
						src = (char*)(data + i*w*h+ j*w);
						dst = (char*)dev_data.ptr + i*dev_data.pitch*dev_data.ysize + j*dev_data.pitch;
						cudaMemcpyAsync(dst,  src, w*sizeof(ushort), cudaMemcpyHostToDevice);
					}
				}
				//sync the device
				cudaMemcpy(dst,  src, w*sizeof(ushort), cudaMemcpyHostToDevice);
			}
			width = w;
			height = h;
			depth = d;
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	bool CopyDeviceToHost(ushort* data, int w, int h, int d){
		assert(w == width && h == height && d == depth);
		bool bRtn = false;
		assert(w == width && h == height && depth == d);
		if (data && dev_data.ptr){
			char* src = NULL;
			char* dst = NULL;
			for (int i= 0; i<d; i ++){
				for (int j=0; j<h; j ++){
					dst = (char*)(data + i*w*h+ j*w);
					src = (char*)dev_data.ptr + i*dev_data.pitch*dev_data.ysize + j*dev_data.pitch;
					cudaMemcpyAsync(dst,  src, w*sizeof(ushort), cudaMemcpyDeviceToHost);
				}
			}
			//sync the device
			cudaMemcpy(dst,  src, w*sizeof(ushort), cudaMemcpyDeviceToHost);

			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	template<class _Tp>
	bool BindToTexture(_Tp& _tex);
	
	cudaPitchedPtr dev_data;
	cudaExtent extent;
	int width, height, depth;
	const int channels; 

};




typedef gpuData<float, 1>    gpuData1dReal32;
typedef gpuData<float, 2>    gpuData2dReal32;
typedef gpuData<double, 2>   gpuData2dReal64;
typedef gpuData<float, 3>    gpuData3dReal32;
typedef gpuData<ushort, 3>   gpuData3dUInt16;
typedef gpuData<float2, 3>   gpuData3dReal32_2;

typedef DataRef<float>       DataRefReal32;
typedef DataRef<double>      DataRefReal64;
typedef DataRef<float2>      DataRefReal32_2;
typedef DataRef<ushort>      DataRefUInt16;








template<typename T> struct DevData2D;

template<> 
struct DevData2D<float>{
	int width, height;
	cudaArray* devPtr;
	cudaChannelFormatDesc channelDesc;
	int pitch;
	DevData2D():devPtr(NULL), width(0), height(0), pitch(0){
	}
	DevData2D(int w, int h):devPtr(NULL), width(0), height(0), pitch(0){
		MallocBuffer(w, h);
	}
	DevData2D& MallocBuffer(int w, int h){
		if (width == w && height == h){
			//reuse the same buffer
		}else{
			channelDesc = cudaCreateChannelDesc<float>();
			cudaMallocArray(&devPtr, &channelDesc, w, h);
			CUDA_CHECK_ERROR;			
			width = w;
			height = h;
		}
		return *this;
	}
	void FreeBuffer(){
		if (devPtr){
			cudaFreeArray((cudaArray*)devPtr);
			CUDA_CHECK_ERROR;
		}
		width = height = 0;
	}
	virtual ~DevData2D(){
		FreeBuffer();
	}
	DevData2D& CopyHostToDevice(const float* data, int w, int h){
		cudaMemcpy2DToArray(devPtr, 0, 0, data, sizeof(float)*w,sizeof(float)*w, h, cudaMemcpyHostToDevice); 
		CUDA_CHECK_ERROR;
		return *this;
	}
	DevData2D& CopyDeviceToHost(float* data, int w, int h){
		cudaMemcpy2DFromArray(data, sizeof(float)*w, devPtr,  0, 0, sizeof(float)*w, h, cudaMemcpyDeviceToHost); 
		CUDA_CHECK_ERROR;
		return *this;
	}
	template<class _Tp>
	DevData2D& BindTexture(_Tp& texref){
		cudaBindTextureToArray(&texref, (cudaArray*)devPtr, &channelDesc);
		CUDA_CHECK_ERROR;
		return *this;
	}
};









#endif //_cudaBackProjection_H
