#ifndef __Type_H
#define __Type_H

//#pragma once

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <ios>
#include <vector>
#include <string>

#include <stdio.h>
#include <iostream>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <vector>
#include <memory>
#include <string>
#include <math.h>
#include <functional>   // std::plus
#include <algorithm>
#include <numeric>

#include <limits.h>
#include <assert.h>
#include <omp.h>

#include <float.h>
#include <math.h>
//#include <thread>
#include <sstream>
#include <limits>
#include <random>
#include <memory>
#include <typeinfo>
#include <chrono>         // std::chrono::milliseconds
#include <list>
#include <mutex>
#include <thread>

#pragma warning(disable : 4244 4267 4101 4309 4305 4819 4018 4799)

#ifdef _WIN32
	#include <windows.h>
	#include <io.h>
	#include <mmsystem.h>
	#include <xmmintrin.h>
	#include <emmintrin.h>
	#include <intrin.h>
	#include <mmsystem.h>
	#include <xmmintrin.h>
	#include <direct.h>
	#pragma comment(lib, "winmm.lib")

	typedef unsigned char       uchar;
	typedef unsigned short      ushort;
	typedef unsigned int        uint;
	typedef unsigned long long  uint64;
	typedef long long           int64;

	#define GetCurrentDir _getcwd
#else
	#include <stdarg.h>
	#include <x86intrin.h>
	#include <unistd.h>
	#include <fcntl.h>
	#include <unistd.h>
	#include <sys/stat.h>
	#include <sys/time.h>

	typedef unsigned char  uchar;
	typedef unsigned short ushort;
	typedef unsigned int   uint;
	typedef uint DWORD;
	typedef long long  __int64;
	typedef long long int64;

#define BOOL bool
#define TRUE true
#define FALSE false

	#define GetCurrentDir getcwd

	#define DeleteFile DeleteFileA
	inline bool DeleteFileA(const char* path){
		return 0 == std::remove(path);
	}
	inline bool CopyFileA(const char* src, const char* dst, bool bSave=true){
		return false;
	}
	inline uint timeGetTime(){
        struct timeval tv;
        if(gettimeofday(&tv, NULL) != 0)
                return 0;
        return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
	}
	inline float timeGetTimefloat() {
		struct timeval tv;
		if (gettimeofday(&tv, NULL) != 0)
			return 0;
		return float(tv.tv_sec * 1000.0) + float(tv.tv_usec / 1000.0);
	}
	inline bool CreateDirectory(const char* szPath, int att){
		mkdir(szPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		return true;
	}
	inline bool CopyFile(const char* src, const char* dst, bool att){
		#ifdef _WIN32
			return CopyFile(src, dst, att)?true:false;
		#else
			bool bRtn = false;
			char buf[1024];
			size_t size;
			FILE* source = fopen(src, "rb");
			FILE* dest = fopen(dst, "wb");
			if (source && dest) {
				bRtn = true;
				while (size = fread(buf, 1, sizeof(buf), source)) {
					if (size != fwrite(buf, 1, size, dest)){
						bRtn = false;
						break;
					}
				}
			}
			if (source) fclose(source);
			if (dest) fclose(dest);
			return bRtn;
		#endif
	}
	#define CreateDirectoryA CreateDirectory
#define _isnan isnan

	#define _access access

#endif

//#define _CVLIB

#ifdef _CVLIB
	#ifdef _WIN32
		#pragma comment(lib, "opencv_highgui249.lib") 
		#pragma comment(lib, "opencv_core249.lib") 
	#endif
	#include "cv.h"
	#include "highgui.h"
	using namespace cv;
#else
	namespace cv{
		typedef void Mat;
	};
#endif



static const double PI  =3.141592653589793238462;

#define DISPLAY_FUNCTION std::cout<<"call_function : "<<__FUNCTION__<<" ......"<<std::endl;

#ifdef max 
#undef max
#endif

#ifdef min
#undef min
#endif

#ifndef MAX
#define MAX(x, y) ((x)>=(y)?(x):(y))
#endif

#ifndef MIN
#define MIN(x, y) ((x)<=(y)?(x):(y))
#endif

//template<typename T> inline
//	T MAX(T x, T y) {
//		return ((x) >= (y) ? (x) : (y));
//}
//
//template<typename T> inline
//	T MIN(T x, T y){
//		return ((x) <= (y) ? (x) : (y));
//}

#define MAX3(x, y, z)  MAX(MAX((x), (y)), (z))
#define MIN3(x, y, z)  MIN(MIN((x), (y)), (z))

#define MAX4(x, y, z, w)  MAX(MAX3((x), (y), (z)), (w))
#define MIN4(x, y, z, w)  MIN(MIN3((x), (y), (z)), (w))

inline std::string hostname() { char szBuf[4096] = ""; gethostname(szBuf, sizeof(szBuf)); return szBuf; }

#define Error_LINE(TXT) StringFormat("Error: %s: %-10s %-10s: %4d: %s\n", hostname().c_str(), __FUNCTION__, __FILE__, __LINE__, TXT)
#define _VERIFY_TRUE(x, _msg) if (!(x)){std::string _m = Error_LINE(#x) + " " + _msg; printf(_m.c_str()); assert(0); throw(std::string(_m)); } 
#define VERIFY_TRUE(x) _VERIFY_TRUE(x, std::string())

#define  FLOAT_EPSINON  0.00001
#define  DOUBLE_EPSINON 1e-10


#define  FLOAT_EPSINON  0.00001
#define  DOUBLE_EPSINON 1e-10

template<typename TSrc, typename TDst> inline TDst Round(const TSrc& x){ return TDst(x); };
template<> inline int Round<float, int>(const float& x){ return x >= 0?(x+0.5):(x-0.5); };
template<> inline short Round<float, short>(const float& x){ return x >= 0?(x+0.5):(x-0.5); };

template<typename T> inline bool IsZero(T value)   { return value == 0; }
template<> inline bool IsZero<float>(float  value) { return value<=FLOAT_EPSINON && value>=-FLOAT_EPSINON; }
template<> inline bool IsZero<double>(double value){ return value<=DOUBLE_EPSINON && value>=-DOUBLE_EPSINON; }

template<typename T> inline bool IsEqual(T x, T y){ return x == y;}
template<> inline bool IsEqual<float>(float x, float y){ return IsZero(x - y);}
template<> inline bool IsEqual<double>(double x, double y){ return IsZero(x - y);}

template<typename TSrc, typename TDst> inline TDst SaturateCast(TSrc v) { return TDst(v); }
template<> inline short SaturateCast<float, short>(float v) { return  (v>SHRT_MAX)?SHRT_MAX:(v<SHRT_MIN?SHRT_MIN:Round<float, short>(v)); }



template<class T> inline bool CvtString( const std::string element, T& val){
		VERIFY_TRUE(0);
		return false;
}

#define DegToRad(x)  ((x)/180.0*PI)
#define RadToDeg(x)  ((x)/PI*180.0)

inline std::string StringFormat(const char* format, ...){
    char szBuf[4096] = "";
	va_list arg;

    va_start(arg, format);
    vsnprintf(szBuf, sizeof(szBuf), format, arg);
    va_end(arg);
    return std::string(szBuf);
}

inline bool IsPathExisted(const char* szPath){
	if (szPath == NULL) return false;
	return _access(szPath, 0) == 0 ? true : false;
}

template<typename T>
	struct BufferT {
		BufferT(int64 len = 0)
			: buffer(NULL)
			, count(len) {
			malloc(len);
		}
		virtual ~BufferT() {
			free();
		}
		inline BufferT& operator=(const BufferT& buf) {
			this->free();
			malloc(buf.size());
			memcpy(this->buffer, buf.data(), buf.size()*sizeof(this->buffer[0]));
			return *this;
		}
		inline bool malloc(size_t len) {
			try {
				buffer = new T[len];
				if (buffer) count = len;
			}
			catch (...) {
				return false;
			}
			return true;
		}
		inline void free() {
			if (buffer) {
				delete[]buffer;
				buffer = NULL;
				count = 0;
			}
		}
		inline BufferT& resize(size_t len) {
			this->free();
			this->malloc(len);
			return *this;
		}
		inline size_t size() const {
			return count;
		}
		inline T* data() const {
			return buffer;
		}
		inline const T& operator[](size_t index) const {
			assert(index >= 0&&index < count);
			return buffer[index];
		}
		inline T& operator[](size_t index) {
			assert(index >= 0 && index < count);
			return buffer[index];
		}
		inline void clear() {
			memset(buffer, 0, sizeof(T)*count);
		}
	private:
		T*  buffer;
		size_t count;
	};

//////////////////////////////////////////////////////////
template<> inline bool CvtString<int>( const std::string element, int& val) 
{
    std::istringstream ios( element);
    ios >> val;
	return ios.eof()?true:false;
}

template<> inline bool CvtString<float>( const std::string element, float& val) {
    std::istringstream ios( element);
    ios >> val;
	return ios.eof()?true:false;
}

template<> inline bool CvtString<double>( const std::string element, double& val){
    std::istringstream ios( element);
    ios >> val;
	return ios.eof()?true:false;
}

template<> inline bool CvtString<std::string>( const std::string element,  std::string& val){
    val = element;
    return true;
}

template<typename _Tp0, typename _Tp1, typename _Tp2> inline
	void FilterStep(const std::vector<_Tp0>& kernel, const _Tp1* src, _Tp2* dst, int width, int stepSrc, int stepDst)
{
	VERIFY_TRUE(kernel.size() > 0);
	int kernel_size = kernel.size();
	int radius = kernel_size / 2;
	std::vector<float> buf(width + 1);
	_Tp1* data = &buf[0];
	const _Tp0* pKernel = &kernel[0];
	for (int i=0; i<width; i ++){
		float sum = 0;
		for (int j=-radius, k=0; j<=radius; j ++, k ++){
			int x = i + j;
			if (x>=0 && x < width)
				sum += src[x*stepSrc]*pKernel[k];
		}
		data[i] = sum;
	}
	for (int i=0; i<width; i ++){
		dst[i*stepDst] = data[i];
	}
}


template<typename T>
struct Point2dT{
	typedef T DataType;
	Point2dT():x(0), y(0){
	}
	template<typename _Tp0, typename _Tp1>
	Point2dT(_Tp0 _x, _Tp1 _y):x(_x), y(_y){
	}
	Point2dT& operator=(const Point2dT& pt){
		x = pt.x;
		y = pt.y;
		return *this;
	}
	Point2dT& operator+=(const Point2dT& pt){
		x += pt.x;
		y += pt.y;
		return *this;
	}
	bool operator ==(const Point2dT& pt) const{
		return (x == pt.x && y == pt.y)?true:false;
	}
	T x, y;
};

typedef Point2dT<int>    Point2D32i;
typedef Point2dT<float>  Point2D32f;
typedef Point2dT<double> Point2D64f;

template<typename T>
struct Point3dT{
	Point3dT():x(0), y(0), z(0){
	}
	template<typename _Tp0, typename _Tp1, typename _Tp2>
	Point3dT(_Tp0 _x , _Tp1 _y , _Tp2 _z ):x(_x), y(_y), z(_z){
	}
	Point3dT& operator=(const Point3dT& pt){
		x = pt.x;
		y = pt.y;
		z = pt.z;
		return *this;
	}
	Point3dT& operator+=(const Point3dT& pt){
		x += pt.x;
		y += pt.y;
		z += pt.z;
		return *this;
	}
	bool operator ==(const Point3dT& pt) const{
		return (x == pt.x && y == pt.y && z == pt.z)?true:false;
	}
	T x, y, z;
};

typedef Point3dT<int>    Point3d32i;
typedef Point3dT<float>  Point3d32f;
typedef Point3dT<double> Point3d64f;

template<typename T> inline
	double Distance(T x0, T y0, T x1, T y1)
{
	double dx = x0 - x1;
	double dy = y0 - y1;
	return sqrt(dx*dx + dy*dy);
}

template<typename _Tp>
struct Size2dT{
	Size2dT(_Tp _cx=0, _Tp _cy=0):cx(_cx), cy(_cy){
	}
	Size2dT(const Size2dT& sz){
		*this = sz;
	}
	inline Size2dT& operator=(const Size2dT& sz){
		cx = sz.cx;
		cy = sz.cy;
		return *this;
	}
	inline bool operator==(const Size2dT& sz) const{
		return (sz.cx == cx && sz.cy == cy)?true:false;
	}
	_Tp cx, cy;
};

typedef Size2dT<int> Size2d32i;
typedef Size2dT<float> Size2d32f;

template<typename _Tp>
struct Size3dT{
	Size3dT(_Tp _cx=0, _Tp _cy=0, _Tp _cz=0):cx(_cx), cy(_cy), cz(_cz){
	}
	Size3dT(const Size3dT& sz){
		*this = sz;
	}
	inline Size3dT& operator=(const Size3dT& sz){
		cx = sz.cx;
		cy = sz.cy;
		cz = sz.cz;
		return *this;
	}
	inline bool operator==(const Size3dT& sz) const{
		return (sz.cx == cx && sz.cy == cy && sz.cz == cz)?true:false;
	}
	_Tp cx, cy, cz;
};

typedef Size3dT<int> Size3d32i;
typedef Size3dT<float> Size3d32f;

template<typename _Tp> 
struct RectT{
	typedef _Tp Value_Type;
	RectT():x(0),y(0),width(0),height(0){
	}
	RectT(_Tp _x, _Tp _y, _Tp _width, _Tp _height):x(_x),y(_y),width(_width),height(_height){
	}
	RectT(const RectT& r):x(r.x),y(r.y),width(r.width),height(r.height){
	}
	inline RectT& operator = ( const RectT& r ){
		x=r.x; y = r.y; width = r.width; height = r.height;
		return *this;
	}
	template<typename T>
	inline bool IsInside(Point2dT<T> pt) const{
		return IsInside(pt.x, pt.y);
	}
	template<typename T>
	inline bool IsInside(T _x, T _y) const{
		return _x>=x && _x<=x+width-1 
			&& _y>=y && _y<=y+height-1;
	}
	inline Size2dT<_Tp> Size() const{
		return Size2dT<_Tp>(width, height);
	}
	inline RectT& Offset(_Tp _x, _Tp _y){
		x += _x; y += _y;
		return *this;
	}
	inline RectT& Offset(Point2dT<_Tp> offset){
		return (*this)(offset.x, offset.y);
	}
	inline _Tp MinX() const{return x;}
	inline _Tp MinY() const{return y;}
	inline _Tp MaxX() const{return x + width - 1;}
	inline _Tp MaxY() const{return y + height - 1;}
	_Tp x, y, width, height;
};

template<typename TA, typename TB> inline
RectT<TA> CalculateOverlap(RectT<TA> boxRef, RectT<TA> boxMove, Point2dT<TB> offset)
{
	RectT<TA> boxNew = boxMove;
	boxNew.Offset(offset.x, offset.y);
	Point2dT<TA> ptMin, ptMax;
	ptMin.x = MAX(boxRef.MinX(), boxNew.MinX());
	ptMin.y = MAX(boxRef.MinY(), boxNew.MinY());
	ptMax.x = MIN(boxRef.MaxX(), boxNew.MaxX());
	ptMax.y = MIN(boxRef.MaxY(), boxNew.MaxY());
	RectT<TA> boxRet(ptMin.x, ptMin.y, ptMax.x - ptMin.x - 1, ptMax.y - ptMin.y - 1);
	return boxRet;
}

typedef RectT<int> Rect32i;

template<typename _Tp>
struct BoxT{
	typedef _Tp DataType;
	BoxT(){
	}
	BoxT(_Tp _x, _Tp _y, _Tp _z, _Tp _width, _Tp _height, _Tp _depth){
		x = _x;
		y = _y;
		z = _z;
		width = _width;
		height = _height;
		depth = _depth;
	}
	inline BoxT& operator=(const BoxT& box){
		x = box.x;
		y = box.y;
		z = box.z;
		width = box.width;
		height = box.height;
		depth = box.depth;
	}
	template<typename T>
	inline bool IsInside(Point3dT<T> pt) const{
		return pt.x>=x && pt.x<=x+width-1 
			&& pt.y>=y && pt.y<=y+height-1 
			&& pt.z>=z && pt.z<=z+depth-1;
	}
	Size3dT<_Tp> Size() const{
		return Size3dT<_Tp>(width, height, depth);
	}
	BoxT& Offset(_Tp _x, _Tp _y, _Tp _z){
		x += _x; y += _y; z += _z;
		return *this;
	}
	BoxT& Offset(Point3dT<_Tp> offset){
		return this->Offset(offset.x, offset.y, offset.z);
	}
	bool IsValid() const{
		return (width > 0 && height > 0 && depth > 0)?TRUE:FALSE;
	}
	_Tp MinX(){return x;}
	_Tp MinY(){return y;}
	_Tp MinZ(){return z;}
	_Tp MaxX(){return x + width - 1;}
	_Tp MaxY(){return y + height - 1;}
	_Tp MaxZ(){return z + depth - 1;}

	_Tp x, y, z, width, height, depth;
};

template<typename TA, typename TB> inline
BoxT<TA> CalculateOverlap(BoxT<TA> boxRef, BoxT<TA> boxMove, Point3dT<TB> offset)
{
	BoxT<TA> boxNew = boxMove;
	boxNew.Offset(offset.x, offset.y, offset.z);
	Point3dT<TA> ptMin, ptMax;
	ptMin.x = MAX(boxRef.MinX(), boxNew.MinX());
	ptMin.y = MAX(boxRef.MinY(), boxNew.MinY());
	ptMin.z = MAX(boxRef.MinZ(), boxNew.MinZ());
	ptMax.x = MIN(boxRef.MaxX(), boxNew.MaxX());
	ptMax.y = MIN(boxRef.MaxY(), boxNew.MaxY());
	ptMax.z = MIN(boxRef.MaxZ(), boxNew.MaxZ());
	BoxT<TA> boxRet(ptMin.x, ptMin.y, ptMin.z, ptMax.x - ptMin.x + 1, ptMax.y - ptMin.y + 1, ptMax.z - ptMin.z + 1);
	return boxRet;
}

typedef BoxT<int> Box32i;
typedef BoxT<float> Box32f;

template<typename T, int DIM_X, int DIM_Y>
struct Matrix2dT{
	typedef T DataType;
	Matrix2dT():dimX(DIM_X), dimY(DIM_Y){
		Zero();
	}
	Matrix2dT(const T* vec, int count):dimX(DIM_X), dimY(DIM_Y){
		assert(count == size());
		memcpy(data, vec, sizeof(T)*count);
	}
	Matrix2dT(const Matrix2dT& m):dimX(DIM_X), dimY(DIM_Y){	
		*this = m;
	}
#define SET_VALUE(i) data[i] = x##i;
	Matrix2dT(T x0, T x1, T x2, T x3)
		:dimX(DIM_X), dimY(DIM_Y){
			assert(size()>=4);
			SET_VALUE(0);  SET_VALUE(1);   SET_VALUE(2);  SET_VALUE(3);
	}
	Matrix2dT(T x0, T x1, T x2, T x3,T x4, T x5, T x6, T x7, T x8)
		:dimX(DIM_X), dimY(DIM_Y){
			assert(size()>=9);
			SET_VALUE(0);  SET_VALUE(1);   SET_VALUE(2);  
			SET_VALUE(3);  SET_VALUE(4);   SET_VALUE(5);   
			SET_VALUE(6);  SET_VALUE(7);   SET_VALUE(8); 
	}
	Matrix2dT(T x0, T x1, T x2, T x3,T x4, T x5, T x6, T x7, T x8, T x9, T x10, T x11,T x12, T x13, T x14, T x15)
		:dimX(DIM_X), dimY(DIM_Y) {
			assert(size()>=16);
			SET_VALUE(0);  SET_VALUE(1);   SET_VALUE(2);  SET_VALUE(3);
			SET_VALUE(4);  SET_VALUE(5);   SET_VALUE(6);  SET_VALUE(7);
			SET_VALUE(8);  SET_VALUE(9);   SET_VALUE(10); SET_VALUE(11);
			SET_VALUE(12); SET_VALUE(13);  SET_VALUE(14); SET_VALUE(15);
	}
	inline Matrix2dT& operator=(const Matrix2dT& m) throw(){
		assert(dimX == m.dimX);
		assert(dimY == m.dimY);
		memcpy(data, m.data, sizeof(data));
		return *this;
	}
	inline const T* row(int y) const throw(){
		return &data[y*dimX];
	}
	inline T* row(int y) throw(){
		return &data[y*dimX];
	}
	inline const T& operator()(int x, int y) const throw(){
		assert(x>=0 && x<dimX);
		return row(y)[x];
	}
	inline T& operator()(int x, int y)  throw(){
		assert(x>=0 && x<dimX);
		return row(y)[x];
	}
	inline void Zero() throw(){
		memset(data, 0, sizeof(data));
	}
	inline int size() const throw(){
		return DIM_X*DIM_Y;
	}
	T trace() const{
		VERIFY_TRUE(dimX == dimY);
		T res(0);
		for (int i=0; i<dimX; i ++){
			res += (*this)(i, i);
		}
		return res;
	}
	inline T determinant () const{
		VERIFY_TRUE(dimX == dimY);
		T res = 0;
		for (int i=0; i<dimX; i ++){
			T sA(1), sB(1);
			int x0(i), y0(0);
			int x1(i), y1(0);
			for (int j=0; j<dimY; j ++, x0 ++, y0 ++, x1 --, y1 ++){
				if (x0 >= dimX) x0 -= dimX;
				if (y0 >= dimY) y0 -= dimY;
				sA *= (*this)(x0, y0);

				if (x1 < 0)     x1 += dimX;
				if (y1 >= dimY) y1 -= dimY;
				sB *= (*this)(x1, y1);
			}
			res += sA - sB;
		}
		return res;

		//return x[0][0]*(x[1][1]*x[2][2] - x[1][2]*x[2][1]) +
		//	x[0][1]*(x[1][2]*x[2][0] - x[1][0]*x[2][2]) +
		//	x[0][2]*(x[1][0]*x[2][1] - x[1][1]*x[2][0]);
	}
	const int dimX, dimY;
#ifdef _WIN32
	__declspec(align(16)) T data[DIM_X*DIM_Y];
#else
	 T data[DIM_X*DIM_Y]  __attribute__ ((aligned (16)));
#endif
};

typedef Matrix2dT<float, 4, 4> Matrix2dReal4x4;
typedef Matrix2dT<float, 4, 3> Matrix2dReal3x4;
typedef Matrix2dT<float, 3, 3> Matrix2dReal3x3;
typedef Matrix2dT<float, 1, 4> Matrix2dReal1x4;

struct Matrix2dReal4x4_SSE{
	inline Matrix2dReal4x4_SSE(const Matrix2dReal4x4& mat){
		mm1 =  _mm_load_ps(mat.data);
		mm2 =  _mm_load_ps(mat.data + 4);
		mm3 =  _mm_load_ps(mat.data + 8);
		mm4 =  _mm_load_ps(mat.data + 12);
	}
	inline void Multiple(const Matrix2dReal1x4& src, Matrix2dReal1x4& dst) const{
		__m128 mm5, mm6, mm7, mm8;
		mm5 =  _mm_load_ps(src.data);

		mm8 = _mm_mul_ps(mm4 , mm5);
		mm7 = _mm_mul_ps(mm3 , mm5);
		mm6 = _mm_mul_ps(mm2 , mm5);
		mm5 = _mm_mul_ps(mm1 , mm5);

		_MM_TRANSPOSE4_PS(mm5, mm6, mm7, mm8);
		mm5 = _mm_add_ps(_mm_add_ps(mm5, mm6), _mm_add_ps(mm7, mm8));
		_mm_store_ps(dst.data, mm5);
	}

	template<bool VALID_X, bool VALID_Y, bool VALID_Z, bool VALID_W>
	inline void MultipleXYZW(const float src[4], float dst[4]) const{
		assert(VALID_Z);
		assert(!VALID_W);
#ifdef _WIN32
		__m128 mm5, mm6, mm7;
		mm5 =  _mm_load_ps(src);

		if (VALID_Z){
			const int MASK_2 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<2);
			mm7 = _mm_dp_ps(mm3, mm5, MASK_2);
		}else{
			mm7 = _mm_setzero_ps();
		}

		if (VALID_X){
			const int MASK_0 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<0);
			mm6 = _mm_dp_ps(mm1, mm5, MASK_0);
			mm7 = _mm_or_ps(mm7, mm6);
		}

		if (VALID_Y){
			const int MASK_1 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<1);
			mm6 = _mm_dp_ps(mm2, mm5, MASK_1);
			mm7 = _mm_or_ps(mm7, mm6);
		}

		if (VALID_W){
			const int MASK_3 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<3);
			mm6 = _mm_dp_ps(mm4, mm5, MASK_3);
			mm7 = _mm_or_ps(mm7, mm6);
		}
#ifdef _WIN32
		mm6 = _mm_set_ps(mm7.m128_f32[3], 1.f, 1.f, 1.f);
#else
#pragma message(TODO)
		assert(0);
#endif
		mm7 = _mm_mul_ps(mm6, mm7);
		 
		_mm_store_ps(dst, mm7);
#endif
	}

	
	template<bool VALID_X, bool VALID_Y, bool VALID_Z, bool VALID_W>
	inline void GetXYZW(const float src[4], float dst[4]) const{
#ifdef _WIN32
		assert(VALID_Z);
		assert(!VALID_W);
		__m128 mm5, mm6, mm7;
		mm5 =  _mm_load_ps(src);
		mm7 = _mm_set_ps(1.f, 1.f, 0, 0);

		if (VALID_X){
			const int MASK_0 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<0);
			mm6 = _mm_dp_ps(mm1, mm5, MASK_0);
			mm7 = _mm_or_ps(mm7, mm6);
		}

		if (VALID_Y){
			const int MASK_1 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<1);
			mm6 = _mm_dp_ps(mm2, mm5, MASK_1);
			mm7 = _mm_or_ps(mm7, mm6);
		}

		if (VALID_Z){
			const int MASK_2 = (1<<4) + (1<<5) + (1<<6) + (1<<7)+ (1<<0)+ (1<<1) + (1<<2) + (1<<3);
			mm6 = _mm_dp_ps(mm3, mm5, MASK_2);
			mm7 = _mm_div_ps(mm7, mm6); 
		}

		_mm_store_ps(dst, mm7);
		//should be improved
		//dst[3] *= dst[3];
#endif		
	}
	__m128 mm1, mm2, mm3, mm4;
};

template<typename T>
inline void MatrixMulti(const T* a, int aw, int ah, const T* b, int bw, int bh, T* c, int cw, int ch)
{
	assert(a && b && c && aw > 0 && ah > 0 && bw > 0 && bh > 0 && cw > 0 && ch > 0);
	assert(aw == bh && ah == ch && bw == cw);
	for (int i=0; i<ch; i ++){
		for (int j=0; j<cw; j++){
			T sum(0);
			for (int k=0; k<aw; k ++){
				sum += a[i*aw+k]*b[j + k*bw];
			}
			c[i*cw+j] = sum;
		}
	}
}


inline Matrix2dReal4x4& MatrixMulti(const Matrix2dReal4x4& m0, const Matrix2dReal4x4& m1, Matrix2dReal4x4& m2)
{
	typedef Matrix2dReal4x4::DataType T;
	MatrixMulti<T>((const T*)m0.data, m0.dimX, m0.dimY, (const T*)m1.data, m1.dimX, m1.dimY, (T*)m2.data, m2.dimX, m2.dimY);
	return m2;
}

inline Matrix2dReal4x4 operator*(const Matrix2dReal4x4& m0, const Matrix2dReal4x4& m1)
{
	Matrix2dReal4x4 m2;
	MatrixMulti(m0, m1, m2);
	return m2;
}


inline Matrix2dReal1x4& MatrixMulti(const Matrix2dReal4x4& m0, const Matrix2dReal1x4& m1, Matrix2dReal1x4& m2)
{
#if 1
	__m128 mm5 =  _mm_load_ps(m1.data);
	__m128 mm1 =  _mm_load_ps(m0.data);
	__m128 mm2 =  _mm_load_ps(m0.data + 4);
	__m128 mm3 =  _mm_load_ps(m0.data + 8);
	__m128 mm4 =  _mm_load_ps(m0.data + 12);

	mm1 = _mm_mul_ps(mm1 , mm5);
	mm2 = _mm_mul_ps(mm2 , mm5);
	mm3 = _mm_mul_ps(mm3 , mm5);
	mm4 = _mm_mul_ps(mm4 , mm5);
#else
	__m128 mm5 =  _mm_load_ps(m1.data);
	__m128* pMM1 = (__m128*)m0.data;
	__m128 mm1 = _mm_mul_ps(pMM1[0] , mm5);
	__m128 mm2 = _mm_mul_ps(pMM1[1] , mm5);
	__m128 mm3 = _mm_mul_ps(pMM1[2] , mm5);
	__m128 mm4 = _mm_mul_ps(pMM1[3] , mm5);
#endif	
	_MM_TRANSPOSE4_PS(mm1, mm2, mm3, mm4);
	mm5 = _mm_add_ps(_mm_add_ps(mm1, mm2), _mm_add_ps(mm3, mm4));
	_mm_store_ps(m2.data, mm5);

	return m2;
}

inline Matrix2dReal1x4& MatrixMultiSSE4(const Matrix2dReal4x4& m0, const Matrix2dReal1x4& m1, Matrix2dReal1x4& m2)
{

	__m128 mm5 =  _mm_load_ps(m1.data);
	__m128 mm1 =  _mm_load_ps(m0.data);
	__m128 mm2 =  _mm_load_ps(m0.data + 4);
	__m128 mm3 =  _mm_load_ps(m0.data + 8);
	__m128 mm4 =  _mm_load_ps(m0.data + 12);
	__m128 mm6, mm7;
#ifdef _WIN32
	const int MASK_0 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<0);
	mm7 = _mm_dp_ps(mm1, mm5, MASK_0);

	const int MASK_1 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<1);
	mm6 = _mm_dp_ps(mm2, mm5, MASK_1);
	mm7 = _mm_or_ps(mm7, mm6);

	const int MASK_2 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<2);
	mm6 = _mm_dp_ps(mm3, mm5, MASK_2);
	mm7 = _mm_or_ps(mm7, mm6);

	const int MASK_3 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<3);
	mm6 = _mm_dp_ps(mm4, mm5, MASK_3);
	mm7 = _mm_or_ps(mm7, mm6);
#else
#pragma message("TODO")
	VERIFY_TRUE(0);
#endif

	_mm_store_ps(m2.data, mm7);

	return m2;
}

template<int DIM_X> inline
	void VecDotSSE4(const float* vecA, const float* vecB, float* vecResult);

#ifdef _WIN32
template<> inline
	void VecDotSSE4<4>(const float* vecA, const float* vecB, float* vecResult)
{
	const int MASK_0 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<0);
	__m128 mm1;
	mm1 = _mm_dp_ps(_mm_load_ps(vecA), _mm_load_ps(vecB), MASK_0);
	_mm_store_ps(vecResult, mm1);
}


template<> inline
	void VecDotSSE4<3>(const float* vecA, const float* vecB, float* vecResult)
{
	const int MASK_0 = (1<<4) + (1<<5) + (1<<6) + (0<<7) + (1<<0);
	__m128 mm1;
	mm1 = _mm_dp_ps(_mm_load_ps(vecA), _mm_load_ps(vecB), MASK_0);
	_mm_store_ps(vecResult, mm1);
}
#endif

inline Matrix2dReal4x4 MakeRotationMatrix(float angleX, float angleY, float angleZ)
{
	float cosx(cos((angleX))), sinx(sin((angleX)));
	float cosy(cos((angleY))), siny(sin((angleY)));
	float cosz(cos((angleZ))), sinz(sin((angleZ)));
	Matrix2dReal4x4 matX(
		1,    0,     0, 0,
		0, cosx, -sinx, 0,
		0, sinx, cosx,  0,
		0,    0,    0,  1);
	Matrix2dReal4x4 matY(
		cosy, 0,     -siny,  0,
		0,    1,     0,  0,
		siny, 0,    cosy,  0,
		0,    0,    0,  1);
	Matrix2dReal4x4 matZ(
		cosz, -sinz, 0,  0,
		sinz, cosz,  0,  0,
		0,       0,  1,  0,
		0,       0,  0,  1);
	Matrix2dReal4x4 mat4 = matX*matY*matZ;
	return mat4;
}

static void TestMatrixMulti()
{	
	typedef Matrix2dReal4x4::DataType T;
	Matrix2dReal4x4 m0;
	Matrix2dReal1x4 m1, m2, m3;

	for (int i=0; i<m0.size(); i ++) m0.data[i] = i + 1;
	for (int i=0; i<m1.size(); i ++) m1.data[i] = i + 1;
	DWORD tm = timeGetTime();
	MatrixMulti<T>((const T*)m0.data, m0.dimX, m0.dimY, (const T*)m1.data, m1.dimX, m1.dimY, (T*)m2.data, m2.dimX, m2.dimY);
	tm = timeGetTime() - tm;
	std::cout<<"time = "<<tm<<"  ";

	tm = timeGetTime();
	MatrixMulti(m0, m1, m3);
	tm = timeGetTime() - tm;
	std::cout<<"time = "<<tm<<"  ";

	tm = timeGetTime();
	MatrixMultiSSE4(m0, m1, m3);
	tm = timeGetTime() - tm;
	std::cout<<"time = "<<tm<<"  ";

	MatrixMulti(m0, m1, m3);
}

template<typename T> inline
	float Mean_Vec(const T* data, size_t size){
		float mean = 0;
		for (int i=0; i<size; i ++) mean += data[i];
		mean /= size;
		return mean;
}


template<typename TA, typename TB, typename TC> inline
	void Add_Vec(const TA* data0, const TB* data1, TC* data2, int size){
#pragma omp parallel for
		for (int i=0; i<size; i++) data2[i] = data0[i] + data1[i];
}

template<> inline
	void Add_Vec<ushort, ushort, ushort>(const ushort* data0, const ushort* data1, ushort* data2, int size){
		const int count = size / (128/16);
		__m128i* p0 = (__m128i*)data0;
		__m128i* p1 = (__m128i*)data1;
		__m128i* p2 = (__m128i*)data2;
#pragma omp parallel for
		for (int i=0; i<count; i ++) {
			p2[i] = _mm_adds_epi16(p0[i], p1[i]);
		}

		for (int i=count*8;i<size; i++) data2[i] = data0[i] + data1[i];
}

template<> inline
	void Add_Vec<float, float, float>(const float* data0, const float* data1, float* data2, int size){
		const int count = size / (4);
		const __m128* p0 = (__m128*)data0;
		const __m128* p1 = (__m128*)data1;
		__m128* p2 = (__m128*)data2;
#pragma omp parallel for
		for (int i=0; i<count; i ++) {
			p2[i] = _mm_add_ps(p0[i], p1[i]);
		}

		for (int i=count*4;i<size; i++) data2[i] = data0[i] + data1[i];
}

template<typename TA, typename TB, typename TC> inline
	void Add_Shift(const TA* data0, const TB shift, TC* data2, int size){
#pragma omp parallel for
		for (int i=0; i<size; i++) data2[i] = data0[i] + shift;
}

template<> inline
	void Add_Shift<float, float, float>(const float* data0, const float data1, float* data2, int size){
		const int count = size / (128/32);
		const __m128* p0 = (__m128*)data0;
		const __m128  mm1 = _mm_set_ps(data1, data1, data1, data1);
		__m128* p2 = (__m128*)data2;
#pragma omp parallel for
		for (int i=0; i<count; i ++) {
			p2[i] = _mm_add_ps(p0[i], mm1);
		}

		for (int i=count*4;i<size; i++) data2[i] = data0[i] + data1;
}


template<typename TA, typename TB, typename TC> inline
	void Sub_Vec(const TA* data0, const TB* data1, TC* data2, int size){
#pragma omp parallel for
		for (int i=0; i<size; i++) data2[i] = data0[i] - data1[i];
}

template<> inline
	void Sub_Vec<float, float, float>(const float* data0, const float* data1, float* data2, int size){
		const int count = size / (128/32);
		const __m128* p0 = (__m128*)data0;
		const __m128* p1 = (__m128*)data1;
		__m128* p2 = (__m128*)data2;
#pragma omp parallel for
		for (int i=0; i<count; i ++) {
			p2[i] = _mm_sub_ps(p0[i], p1[i]);
		}

		for (int i=count*4;i<size; i++) data2[i] = data0[i] + data1[i];
}


template<typename TA, typename TB, typename TC> inline
	void Sub_Shift(const TA* data0, const TB shift, TC* data2, int size){
		Add_Shift<TA, TB, TC>(data0, -shift, data2, size);
}

template<> inline
	void Sub_Shift<float, float, float>(const float* data0, const float shift, float* data2, int size){
		Add_Shift(data0, -shift, data2, size);
}

template<typename TA, typename TB, typename TC> inline
	void Multiple_Vec(const TA* data0, const TB* data1, TC* data2, int size){
#pragma omp parallel for
		for (int i=0; i<size; i++) data2[i] = data0[i] * data1[i];
}

template<> inline
	void Multiple_Vec<float, float, float>(const float* data0, const float* data1, float* data2, int size){
		const int count = size / (128/32);
		const __m128* p0 = (__m128*)data0;
		const __m128* p1 = (__m128*)data1;
		__m128* p2 = (__m128*)data2;
#pragma omp parallel for
		for (int i=0; i<count; i ++) {
			p2[i] = _mm_mul_ps(p0[i], p1[i]);
		}

		for (int i=count*4;i<size; i++) data2[i] = data0[i] + data1[i];
}

template<typename TA, typename TB, typename TC> inline
	void Multiple_Scale(const TA* data0, const TB scale, TC* data2, int size){
#pragma omp parallel for
		for (int i=0; i<size; i++) data2[i] = data0[i] * scale;
}

template<> inline
	void Multiple_Scale<float, float, float>(const float* data0, const float scale, float* data2, int size){
		const int count = size / (128/32);
		const __m128* p0 = (__m128*)data0;
		const __m128  mm1 = _mm_set_ps(scale, scale, scale, scale);
		__m128* p2 = (__m128*)data2;
#pragma omp parallel for
		for (int i=0; i<count; i ++) {
			p2[i] = _mm_mul_ps(p0[i], mm1);
		}

		for (int i=count*4;i<size; i++) data2[i] = data0[i] * scale;
}

template<typename TA, typename TB, typename TC> inline
	void Divide_Vec(const TA* data0, const TB* data1, TC* data2, int size){
#pragma omp parallel for
		for (int i=0; i<size; i++){
			if (IsZero(data1[i])) data2[i] = 0;
			else                  data2[i] = data0[i] / data1[i];
		}
}

template<typename TA, typename TB, typename TC> inline
	void Divide_Scale(const TA* data0, const TB scale, TC* data2, int size){
		if (IsZero(scale)) memset(data2, 0, sizeof(data2[0])*size);
		else Multiple_Scale(data0, float(1.0)/scale, data2, size);
}

//////////////////////////////////////////////////////////////////////////////
inline void ClearCurrentLine(){
	printf("\r                                                         \r");
}

template<typename T> inline
	bool isNan(T f){
		return false;
}

template<> inline bool isNan<double>(double f){
#ifdef _WIN32
	return _isnan(f)?true:false;
#else
	return std::isnan(f)?true:false;
#endif
}

#if defined(_M_X64)
template<> inline bool isNan<float>(float f){
	return _isnanf(f)?true:false;
}
#endif

//////////////////////////begin opencv function
inline int WaitKey(int key=0){
#ifdef _CVLIB
	return cvWaitKey(key);
#else
	return 0;
#endif

}

template<typename T> inline
cv::Mat ShowImage(std::string name, const T* data, int width, int height, T nMax=T(0), T nMin=T(0), const Rect32i* roi = NULL)
{
#ifdef _CVLIB
#if 0
#pragma message("do not showimage")
	return;
#endif
	//cv::Mat mat;
	IplImage* img = NULL;
	int i, j;
	if (roi){
		Rect32i rc;
		rc.x = std::max(roi->x, 0);
		rc.y = std::max(roi->y, 0);
		rc.width = std::min(roi->width, roi->x + roi->width - rc.x);
		rc.height = std::min(roi->height, roi->y + roi->height - rc.y);
		img = cvCreateImage(cvSize(rc.width, rc.height), IPL_DEPTH_8U, 1);

		if (nMax == nMin){
			nMax = -99999999; //std::max(buffer, buffer+width*height, 0xFFF);
			nMin = +99999999; //std::max(buffer, buffer+width*height, -0xFFF);	
			for (i=rc.y; i<rc.y+rc.height; i ++){
				for (j=rc.x; j<rc.x+rc.width; j++){
					if (isNan(data[i*width+j])) continue;
					//if (_isnanf(data[i*width+j])) continue;
					//if (_isnan(data[i*width+j])) continue;
					nMax = MAX(data[i*width+j], nMax);
					nMin = MIN(data[i*width+j], nMin);
				}
			}

			for (i=0; i<rc.height; i++){
				for (j=0; j<rc.width; j++){
					T val = (*(data + (i+rc.y)*width + j+rc.x) - nMin)*0xff/float(nMax - nMin+0.00001);
					CV_IMAGE_ELEM(img, unsigned char, i, j) = MAX(MIN(val, 0xff), 0);
				}
			}

		}

	}else{
		img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		if (nMax == nMin){
			nMax = data[0]; //std::max(buffer, buffer+width*height, 0xFFF);
			nMin = data[0]; //std::max(buffer, buffer+width*height, -0xFFF);
			for (i=1; i<width*height; i++){
				if (data[i] > nMax) nMax = data[i];
				if (data[i] < nMin) nMin = data[i];
			}
		}

		for (i=0; i<height; i++){
			for (j=0; j<width; j++){
				T val = (*(data + i*width + j) - nMin)*0xff/float(nMax - nMin + 0.00001);
				CV_IMAGE_ELEM(img, unsigned char, i, j) = MAX(MIN(val, 0xff), 0);
			}
		}
	}
	//IplImage* imgScale = cvCreateImage(cvSize(img->width/16, img->height/16), IPL_DEPTH_8U, 1);
	//cvResize(img, imgScale);
	IplImage* colImg = cvCreateImage(cvGetSize(img), img->depth, 3);
	for (int i=0; i<img->height; i ++){
		unsigned char* pCol = (unsigned char*)colImg->imageData + i*colImg->widthStep;
		unsigned char* pImg = (unsigned char*)img->imageData + i*img->widthStep;
		for (int j=0; j<img->width; j ++, pCol += 3, pImg ++){
			//pCol[j*3+0] = pCol[j*3+1] = pCol[j*3+2] = pImg[j];
			pCol[0] = pCol[1] = pCol[2] = pImg[0];
		}
	}

	cvNamedWindow(name.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(name.c_str(), colImg);
	//cvSaveImage("c:/cur_img.bmp", img);
	//cvShowImage(name.c_str(), imgScale);
	WaitKey(5);

	cv::Mat mat(colImg, true);
	if (img) cvReleaseImage(&img);
	if (colImg) cvReleaseImage(&colImg);
	//if (imgScale) cvReleaseImage(&imgScale);
	return mat;
#else

#endif
}

template<typename T>  inline
void SaveImage8U(std::string path, T* data, int width, int height, int nMax=0, int nMin=0)
{
#ifdef _CVLIB
	cv::Mat mat;
	mat.create(cv::Size(width, height), CV_8U);
	int i, j;
	if (nMax == nMin){
		nMax = data[0]; //std::max(buffer, buffer+width*height, 0xFFF);
		nMin = data[0]; //std::max(buffer, buffer+width*height, -0xFFF);
		for (i=1; i<width*height; i++){
			if (data[i] > nMax) nMax = data[i];
			if (data[i] < nMin) nMin = data[i];
		}
	}

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			mat.at<uchar>(i, j) = (*(data + i*width + j) - nMin)*0xff/float(nMax - nMin);
		}
	}
	cv::imwrite(path, mat);
#endif
}

template<typename T> inline
void SaveImage16U(std::string path, T* data, int width, int height, int nMax=0, int nMin=0)
{
#ifdef _CVLIB
	cv::Mat mat;
	mat.create(cv::Size(width, height), CV_16U);
	int i, j;
	if (nMax == nMin){
		nMax = data[0]; //std::max(buffer, buffer+width*height, 0xFFF);
		nMin = data[0]; //std::max(buffer, buffer+width*height, -0xFFF);
		for (i=1; i<width*height; i++){
			if (data[i] > nMax) nMax = data[i];
			if (data[i] < nMin) nMin = data[i];
		}
	}

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			mat.at<ushort>(i, j) = (*(data + i*width + j) - nMin)*0xfff/float(nMax - nMin);
		}
	}
	cv::imwrite(path, mat);
#endif
}

template<typename T> inline
void SaveImageRaw16U(std::string path, T* data, int width, int height, int nMax=0, int nMin=0)
{
#ifdef _CVLIB
	cv::Mat mat;
	mat.create(cv::Size(width, height), CV_16U);
	int i, j;
	if (nMax == nMin){
		nMax = data[0]; //std::max(buffer, buffer+width*height, 0xFFF);
		nMin = data[0]; //std::max(buffer, buffer+width*height, -0xFFF);
		for (i=1; i<width*height; i++){
			if (data[i] > nMax) nMax = data[i];
			if (data[i] < nMin) nMin = data[i];
		}
	}

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			mat.at<ushort>(i, j) = (*(data + i*width + j) - nMin)*0xfff/float(nMax - nMin);
		}
	}
	FILE* fp = fopen(path.c_str(), "wb");
	if (fp){
		fwrite(mat.data, sizeof(ushort), mat.size().width*mat.size().height, fp);
		fclose(fp);
	}
#endif
}

inline void DestroyAllWindow(){
#ifdef _CVLIB
	::cvDestroyAllWindows(); 
#endif
}

inline void DestroyWindow(std::string name){
#ifdef _CVLIB
	::cvDestroyWindow(name.c_str()); 
#endif	
}



//end opencv function
///////////////////////////////////

inline std::string TrimLeft(std::string str, std::string val) {
	if (!str.empty()) {
		for (;;) {
			std::size_t pos = str.find_first_of(val);
			if (pos == 0) {
				str.erase(pos, val.size());
			}
			else {
				break;
			}
		}
	}
	return str;
}

inline std::string TrimRight(std::string str, std::string val) {
	if (!str.empty()) {
		for (std::size_t pos = 0; pos != std::string::npos;) {
			pos = str.find_last_of(val);
			if (pos + val.size() == str.size()) {
				str.erase(pos, val.size());
			}
			else {
				break;
			}
		}
	}
	return str;
}

inline void MakePathDirIfNeed(const char* sz){
	char szTmp[1024*5] = "";
	const char* src = sz;
	const char* p = 0;
	int nTotal = strlen(sz);
	for (;;){
		p = strchr(src, '\\');
		if (!p) p = strchr(src, '/');
		if (p){
			int len =  nTotal - strlen(p);
			if (len > 0){
				strncpy(szTmp, sz, len);
				if (_access(szTmp, 0) != 0){
#ifdef _WIN32
					::CreateDirectoryA(szTmp, 0);
#else
					CreateDirectory(szTmp, 0);
#endif
				}
			}
			src = ++p;
		}else break;
	}
}

//caculate the file size
inline long FileSize(FILE* fp) {
	if (fp == 0) return 0;
	long cur = ftell(fp);
	fseek(fp, 0, SEEK_END);
	long nsize = ftell(fp);
	fseek(fp, cur, SEEK_SET);
	return nsize;
}	

//caculate the file size
inline long FileSize(const char* pszPath) {
	FILE* fp = fopen(pszPath, "rb");
	if (fp == 0) return 0;
	long nsize = FileSize(fp);
	fclose(fp);
	return nsize;
}


template<typename TSrc, typename TDst>
bool WriteToFile(std::string path, const TSrc* data, int count, bool show = false)
{
	bool bRtn = false;
	std::vector<TDst> buf(count+1);
	TDst* pDst = &buf[0];
	for (int i=0; i<count; i++){
		pDst[i] = SaturateCast<TSrc, TDst>(data[i]);
	}
	MakePathDirIfNeed(path.c_str());
	FILE* fp = fopen(path.c_str(), "wb");
	if (fp){
		if (count == fwrite(pDst, sizeof(*pDst), count, fp)){
			bRtn = true;
		}
		fclose(fp);
	}

	if (show)
		printf("WriteToFile : %s, %s\n", path.c_str(), bRtn ? "successed" : "failed");

	return bRtn;
}

//y = ax + b
//linear least mean square fitting
//////////////////////////////////
template<typename _Tp0, typename _Tp1, typename T>
void LinearLeastMeanSquare(const _Tp0* data_x, const _Tp1* data_y, int size, T& a, T& b)
{
	double mean_x = 0;
	double mean_y = 0;
	double A(0), B(0);
	{
#pragma omp parallel for reduction(+:mean_x, mean_y) 
		for (int i=0; i<size; i++){
			mean_x += data_x[i];
			mean_y += data_y[i];
		}
	}
	mean_x /= size;
	mean_y /= size;

	{
#pragma omp parallel for reduction(+:A, B) 
		for (int i=0; i<size; i++){
			T m = data_x[i] - mean_x;
			T n = data_y[i]- mean_y;
			A += m*n;
			B += m*m;
		}
	}
	a = IsZero(B)?0:(A/B);
	b = mean_y - a*mean_x;
}

static void TestLMS()
{
	float x[] = {208, 152, 113, 227, 137, 238, 178, 104, 191, 130,};
	float y[] = {21.6, 15.5, 10.4, 31.0, 13.0, 32.4, 19.0, 10.4, 19.0, 11.8,};
	float a, b;
	LinearLeastMeanSquare(x, y, sizeof(x)/sizeof(x[0]), a, b);
	float sum = 0;
	for (int i=0; i<10; i ++){
		sum += y[i] - a*x[i] - b;
	}
}

template<typename T> inline
	T Length(const T& x, const T& y){
		return sqrt(x * x + y * y);
}

template<typename T>
struct Vector3{
	T x, y, z;

	Vector3(T x=0, T y=0, T z=0):x(x),y(y),z(z){
	}

	Vector3(const Vector3& vec){
		x = vec.x; y = vec.y; z = vec.z;
	}

	Vector3& operator=(const Vector3& vec){
		x = vec.x; y = vec.y; z = vec.z;
		return *this;
	}

	inline T Dot(const Vector3& vec) const{
		return x*vec.x + y*vec.y + z*vec.z;
	}
	inline double Length() const{
		return sqrt(double(x*x + y*y + z*z));
	}
	inline Vector3<T>& Normize(){
		double s = Length();
		x /= s; y /= s; z /= s;
		return *this;
	}

	inline Vector3<T>& Scale(T s){
		x *= s; y *= s; z *= s;
		return *this;
	}

	template<typename D> inline
		static Vector3<T> make(D a){
			Vector3<T> vec;
			vec.x = vec.y = vec.z = T(a);
			return vec;
	}

	template<typename D> inline
		static Vector3<D> Min(const Vector3<D>& a, const Vector3<D>& b){
			Vector3<D> vec(MIN(a.x, b.x), MIN(a.y, b.y), MIN(a.z, b.z));
			return vec;
	}
	template<typename D> inline
		static Vector3<D> Max(const Vector3<D>& a, const Vector3<D>& b){
			Vector3<D> vec(MAX(a.x, b.x), MAX(a.y, b.y), MAX(a.z, b.z));
			return vec;
	}
};

template<typename T> 
inline Vector3<T> operator+(const Vector3<T>& lval, const Vector3<T>& rval)
{
	return Vector3<T>(lval.x + rval.x, lval.y + rval.y, lval.z + rval.z);
}

template<typename T>
inline Vector3<T> operator-(const Vector3<T>& lval, const Vector3<T>& rval)
{
	return Vector3<T>(lval.x - rval.x, lval.y - rval.y, lval.z - rval.z);
}



template<typename T> 
inline Vector3<T> operator*(const Vector3<T>& lval, const Vector3<T>& rval)
{
	return Vector3<T>(lval.x * rval.x, lval.y * rval.y, lval.z * rval.z);
}

template<typename T> inline
	Vector3<T> operator/(const Vector3<T>& lval, const Vector3<T>& rval)
{
	assert(0 != rval.x && 0 != rval.y && 0 != rval.z);
	T x, y, z;
	x = y = z = std::numeric_limits<T>::max();
	if (rval.x != 0) x = lval.x / rval.x;
	if (rval.y != 0) y = lval.y / rval.y;
	if (rval.z != 0) z = lval.z / rval.z;
	return Vector3<T>(x, y, z);
}

template<typename T, typename A>
inline Vector3<T> operator*(const Vector3<T>& lval, A s)
{
	return Vector3<T>(lval.x * s, lval.y * s, lval.z * s);
}

template<typename T, typename A>
inline Vector3<T> operator/(const Vector3<T>& lval, A s)
{
	assert(!IsZero(s));
	static const T vMAX = std::numeric_limits<T>::max();
	if (s == 0) return Vector3<T>(vMAX, vMAX, vMAX);
	return Vector3<T>(lval.x / s, lval.y / s, lval.z / s);
}

template<typename T, typename A>
inline Vector3<T> operator+(const Vector3<T>& lval, A s)
{
	return Vector3<T>(lval.x + s, lval.y + s, lval.z + s);
}

template<typename T, typename A>
inline Vector3<T> operator-(const Vector3<T>& lval, A s)
{
	return Vector3<T>(lval.x - s, lval.y - s, lval.z - s);
}


typedef Vector3<float> Float3;

template<typename T>
struct RayT{
	Vector3<T> o;   // origin
	Vector3<T> d;   // direction
};

typedef RayT<float> Ray;

template<typename T>
inline bool IntersectBox(const RayT<T>& r,const Vector3<T>& boxmin,const Vector3<T>& boxmax, T *tnear, T *tfar)
{
	// compute intersection of ray with all six bbox planes
	//float3 invR = make_float3(1.0f) / r.d;
	Vector3<T> invR = Vector3<T>::make(1.0f) / r.d;
	Vector3<T> tbot = invR * (boxmin - r.o);
	Vector3<T> ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	//float3 tmin = fminf(ttop, tbot);
	Vector3<T> tmin = Vector3<T>::Min(ttop, tbot);
	Vector3<T> tmax = Vector3<T>::Max(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = MAX(MAX(tmin.x, tmin.y), MAX(tmin.x, tmin.z));
	float smallest_tmax = MIN(MIN(tmax.x, tmax.y), MIN(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin?true:false;
}
//
///////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool LoadFromFile(const char* szPath, T* buffer, int64 size, bool show=false){
	bool brtn = false;
	FILE* fp = fopen(szPath, "rb");
	if (fp){
		int64 nRead = fread(buffer, sizeof(*buffer), size, fp);
		assert(nRead == size);
		fclose(fp);
		brtn = true;
	}
	if (show)
		printf("load : %s, %s\n", szPath, brtn ? "successed" : "failed");
	return brtn;
}

template<typename T>
bool LoadImageFromFile(const char* szPath, T* buffer, int width, int height, int depth = 1){
	return LoadFromFile(szPath, buffer, width*height*depth, false);
}

template<typename T>
inline void _GetMaxMin(const T* data, int size, T& valMax, T& valMin, int* pIdxMax = NULL, int* pIdxMin = NULL)
{
#ifdef _WIN32
	if (!data) return;
	if (size <= 0) return;
	int idxMax, idxMin;
	valMax = valMin = data[0];
	idxMax = idxMin = 0;

	const int BLOCK_COUNT = 512*512;
	const int THREAD_COUNT = std::max(std::min(std::max(4, omp_get_num_threads()), size/BLOCK_COUNT), 1);;
	std::vector<int> vecIdxMax(THREAD_COUNT+1);
	std::vector<int> vecIdxMin(THREAD_COUNT+1);
	int* _idxMax = &vecIdxMax[0];
	int* _idxMin = &vecIdxMin[0];
	{
		#pragma omp parallel for num_threads(THREAD_COUNT)
		for (int i=0; i<size; i++){
			int tid = omp_get_thread_num();
			if (data[_idxMax[tid]] < data[i]) _idxMax[tid] = i;
			if (data[_idxMin[tid]] > data[i]) _idxMin[tid] = i;
		}
	}
	{
		int iMax = 0;
		int iMin = 0;
		for (int i=1; i<THREAD_COUNT; i ++){
			if (data[_idxMax[iMax]] < data[_idxMax[i]])  iMax = i;
			if (data[_idxMin[iMin]] > data[_idxMin[i]])  iMin = i;
		}
		idxMax = _idxMax[iMax];
		idxMin = _idxMin[iMin];
	}

	valMax = data[idxMax];
	valMin = data[idxMin];

	if (pIdxMax) *pIdxMax = idxMax;
	if (pIdxMin) *pIdxMin = idxMin;
#else
	VERIFY_TRUE(0);
#endif
}

template<typename T>
inline void GetMaxMin(const T* data, int size, T& valMax, T& valMin, int* pIdxMax = NULL, int* pIdxMin = NULL)
{
	if (!data) return;
	if (size <= 0) return;
	int idxMax, idxMin;
	valMax = valMin = data[0];
	idxMax = idxMin = 0;
	for (int i=1; i<size; i++){
		if (data[idxMax] < data[i]) idxMax = i;
		if (data[idxMin] > data[i]) idxMin = i;
	}
	valMax = data[idxMax];
	valMin = data[idxMin];
	if (pIdxMax) *pIdxMax = idxMax;
	if (pIdxMin) *pIdxMin = idxMin;
}

static void TestMaxMinMT()
{
	for (int i=0; i<10000; i ++){
		std::vector<int> A(512*512*512);
		//std::random_shuffle(A.begin(), A.end());
		//std::fill(A.begin(), A.end(), std::rand());
		std::generate(A.begin(), A.end(), std::rand);
		int iMax0, iMin0;
		int idxMax0, idxMin0;
		int iMax1, iMin1;
		int idxMax1, idxMin1;
		DWORD tm0, tm1, tm2;
		tm0 = timeGetTime();
		_GetMaxMin(&A[0], A.size(), iMax0, iMin0, &idxMax0, &idxMin0);
		tm1 = timeGetTime();
		GetMaxMin(&A[0], A.size(), iMax1, iMin1, &idxMax1, &idxMin1);
		tm2 = timeGetTime();
		VERIFY_TRUE(idxMax0 == idxMax1 && idxMin0 == idxMin1);
		VERIFY_TRUE(iMax0 == iMax1 && iMin0 == iMin1);
		std::cout<<i<<"mt_tm = "<<tm1 - tm0<<" st_tm = "<<tm2 - tm1<<std::endl;
	}
}

template<typename TSrc, typename TDst> inline
	bool Transform(const TSrc* src, TDst* dst, int size)
{
	if (src && dst){
		for (int i=0; i<size; i ++){
			dst[i] = SaturateCast<TSrc, TDst>(src[i]);
		}
		return true;
	}else{
		return false;
	}
}

template<> inline
bool  Transform<float, unsigned char>(const float* src, unsigned char* dst, int size)
{
	if (src && dst){
		float tMax, tMin;
		GetMaxMin(src, size, tMax, tMin);
		for (int i=0; i<size; i ++){
			if (tMax- tMin > 0) dst[i] = (src[i] - tMin)/(tMax- tMin)*255.0;
			else                dst[i] = 0;
		}
		return true;
	}else{
		return false;
	}
}

template<typename _Tp> inline
bool SaveDataDimensionN(std::string path, const _Tp* buffer, int nx, int ny, int nz){
	bool bRtn = false;
	::MakePathDirIfNeed(path.c_str());
	FILE* fp = fopen(path.c_str(), "wb");
	if (fp){
		fwrite(&nx, sizeof(nx), 1, fp);
		fwrite(&ny, sizeof(ny), 1, fp);
		fwrite(&nz, sizeof(nz), 1, fp);
		const int size = nx*ny*nz;
		if (size == fwrite(buffer, sizeof(*buffer), size, fp)) 
			bRtn = true;
		fclose(fp);	
	}	
	printf("%s, SaveDataDimension3:%s\n", bRtn?"successed":"failed", path.c_str());
	return bRtn;
}

template<typename T>
inline void ShiftToNoneNegative(T* data, int size)
{
	T valMax, valMin;
	GetMaxMin<T>(data, size, valMax, valMin);
#pragma omp parallel for num_threads(4)
	for (int i=0; i<size; i ++) data[i] -= valMin;
}

template<typename T>
void Reverse(T* data){
	unsigned char* p = (unsigned char*)data;
	const int Len = sizeof(data[0]);
	for (int i=0; i<Len/2; i ++){
		unsigned char a = p[i];
		p[i] = p[Len-1-i];
		p[Len-1-i] = a;
	}
}

template<typename TSrc, typename TDst>
bool LoadProjectionImageFolder(const char* szFolder, int width, int height, std::vector<TDst*> vec_raw, int step = 1, int _startIndex = 0)
{
	bool bRtn = true;
	const int size = width*height;
	std::vector<std::string> vecPath;
	for (int i=0; i<vec_raw.size(); i += step){
		char szPath[1024];
		sprintf(szPath, "%simg%04d.raw", szFolder, i + _startIndex);	
		vecPath.push_back(szPath);
	}
	for (int i=vecPath.size()-1; i>=0; i--){
		if (FileSize(vecPath[i].c_str()) != sizeof(TSrc)*size){
			return false;
		}
	}
	const int count = vecPath.size();

#pragma omp parallel for num_threads(4)
	for (int i=0; i<count; i++){
		std::vector<TSrc> tmp(size+16);
		TSrc* buf = &tmp[0];
		bRtn = LoadImageFromFile<TSrc>(vecPath[i].c_str(), buf, width, height);
		if (bRtn)
			for (int k=0; k<size; k++){
				//Reverse((unsigned short*)&buf[k]);
				vec_raw[i*step][k] = buf[k];
			}
			assert(bRtn);	
	}

	return bRtn;
}

template<typename T>
std::vector<T> MinusFlag(const std::vector<T>& val)
{
	std::vector<double> _vecUs = val;
	std::vector<double> _vecZero(val.size());
	std::transform(_vecZero.begin(), _vecZero.end(),  val.begin(), _vecUs.begin(), std::minus<double>());
	return _vecUs;
}

template<typename T>
struct Matrix2d{
	Matrix2d(){
	}
	Matrix2d(int w, int h){
		Set(w, h);
	}
	Matrix2d& Transpose(){
		Matrix2d mat(height, width);
		for (int i=0; i<height; i++){
			for (int j=0; j<width; j++){
				mat(i, j) = Get(j, i);
			}
		}
		*this = mat;
		return *this;
	}
	Matrix2d& operator=(const Matrix2d& m){
		Set(m.width, m.height);
		this->data = m.data;
		return *this;
	}
	void Set(int w, int h){
		width=w; height=h;
		data.resize(w*h);	
	}

	T& Get(int x, int y){
		assert(x>=0 && x<width && y>=0 && y<height);
		return data[y*width+x];
	}
	T& operator()(int x, int y){
		return Get(x, y);
	}
	int width, height;
	std::vector<T> data;
};

template<typename T>
void MeshGrid(const std::vector<T>& x, const std::vector<T>& y, Matrix2d<T>& X, Matrix2d<T>& Y)
{
	int width  = x.size();
	int height = y.size();
	X.Set(width, height);
	Y.Set(width, height);
	for (int i=0; i<height; i ++){
		for (int j=0; j<width; j++){
			X(j, i) = x[j];
			Y(j, i) = y[i];
		}
	}
}

template<typename D>
struct FrameT{
	typedef D DataType;
	FrameT(D* data, int width, int height, int index, Rect32i* roi=NULL)
		:buffer(data),width(width),height(height),index(index), roi(roi){
	}
	FrameT(const FrameT& obj)
		:buffer(0),width(0),height(0),index(0), roi(0){
			*this = obj;
	}
	FrameT(){
		memset(this, 0, sizeof(*this));
		index = -1;
	}
	FrameT& operator=(const FrameT& obj){
		buffer = obj.buffer;
		width = obj.width;
		height = obj.height;
		index = obj.index;
		roi = obj.roi;
		return *this;
	}
	const D& operator[](int x) const{
		return buffer[x];
	}
	inline const D& operator()(int x, int y) const{
		assert(x>=0 && x < width && y >= 0 && y < height);
		return buffer[y*width+x];
	}
	inline D& operator()(int x, int y) {
		assert(x>=0 && x < width && y >= 0 && y < height);
		return buffer[y*width+x];
	}
	inline D interp(float x, float y) const{
		return interp_nearest(x+0.5, y+0.5);
	}
	inline D interp_nearest(int x, int y) const{
		D val = 0;
		if (x >= 0 && x < width && y>= 0 && y<height){
			if (roi){
				if (x>= roi->x && x < roi->x + roi->width && y >= roi->y && y < roi->y + roi->height){
					val = buffer[y*width+x];
				}
			}else{
				val = buffer[y*width+x];
			}
		}
		return val;
	}
	inline bool GetRoiData(FrameT& frame, int left, int top, int right, int bottom){
		bool bRtn = false;
		if (right - left + 1 == frame.width && bottom - top + 1 == frame.height && frame.buffer){
			bRtn = true;
			DataType* buf0 = buffer+ width*top + left;
			DataType* buf1 = frame.buffer;
			for (int i=0; i<frame.height; i ++, buf0 += width, buf1 += frame.width){
				memcpy(buf1, buf0, frame.width*sizeof(DataType));
			}
		}
		return bRtn;
	}
	inline FrameT& Ones(){
		if (buffer){
			int size = width*height;
			for (int i=0; i<size; i ++) buffer[i] = D(1);
		}
		return *this;
	}
	inline FrameT& Zero(){
		if (buffer){
			memset(buffer, 0, sizeof(buffer[0])*width*height);
		}
		return *this;
	}
	inline bool SetData(const D* _buffer, int w, int h){
		bool bRtn = false;
		if (buffer && width == w && height == h){
			bRtn = true;
			int size = w*h;
			for (int i=0; i<size; i++) this->buffer[i] = _buffer[i];
		}
		return bRtn;
	}
	void display(std::string name, const Rect32i* curRoi = NULL) const
	{	
		D valMax = 0; //std::max(buffer, buffer+width*height, 0xFFF);
		D valMin = 0; //std::max(buffer, buffer+width*height, -0xFFF);
		::ShowImage(name, this->buffer, width, height, valMax, valMin, curRoi==NULL?this->roi:curRoi);
	}
	FrameT& operator-=(const FrameT& frm){
		if (buffer && frm.buffer && width == frm.width && height == frm.height){
			int size = width*height;
			for (int i=0; i<size; i++) this->buffer[i] -= frm.buffer[i];
		}
		return *this;
	}
	FrameT& operator+=(const FrameT& frm){
		if (buffer && frm.buffer && width == frm.width && height == frm.height){
			int size = width*height;
			for (int i=0; i<size; i++) this->buffer[i] += frm.buffer[i];
		}
		return *this;
	}
	FrameT& Sacle(float val){
		if (buffer){
			int size = width*height;
			for (int i=0; i<size; i++) this->buffer[i] *= val;
		}
		return *this;
	}
public:
	Rect32i* roi;
	D* buffer;
	int width, height;
	int index;
};

template<typename T>
struct ImageT{
	typedef T DataType;
	ImageT(const FrameT<T>& frame):width(0),height(0),buffer(NULL),roi(NULL){
		MallocBuffer(frame.width, frame.height);
		const int size = width*height;
#pragma omp parallel for
		for (int i=0; i<size; i++){
			buffer[i] = frame.buffer[i];
		}
		if (frame.roi) this->SetRoi(*frame.roi);
	}
	template<typename TypeA>
	ImageT(const TypeA* data, int w, int h):width(0),height(0),buffer(NULL),roi(NULL){
		MallocBuffer(w, h);
		const int size = width*height;
#pragma omp parallel for
		for (int i=0; i<size; i++){
			buffer[i] = data[i];
		}
	}
	ImageT(const ImageT& img):width(0),height(0),buffer(NULL),roi(NULL){
		MallocBuffer(img.width, img.height);
		const int size = width*height;
#pragma omp parallel for
		for (int i=0; i<size; i++){
			buffer[i] = img.buffer[i];
		}
		if (img.roi) this->SetRoi(*img.roi);
	}
	ImageT(int w, int h):width(0),height(0),buffer(NULL),roi(NULL){
		MallocBuffer(w, h);
	}
	ImageT():width(0),height(0),buffer(NULL),roi(NULL){
	}
	//ImageT(const ImageT& img):width(0),height(0),buffer(NULL),roi(NULL){
	//	*this = img;
	//}
	ImageT& operator=(const ImageT& img){
		MallocBuffer(img.width, img.height);
		const int size = width*height;
		#pragma omp parallel for
		for (int i=0; i<size; i++){
			buffer[i] = img.buffer[i];
		}
		if (img.roi) this->SetRoi(*img.roi);
		return *this;
	}
	ImageT& Zero(){	
		if (buffer){
			VERIFY_TRUE(buffer == &mem[0]);
			memset(buffer, 0, sizeof(buffer[0])*width*height);
		}
		return *this;
	}
	ImageT& Ones(){
		SetValue(1);
		return *this;
	}
	template<typename D>
	ImageT& SetValue(D val){
		int size = width*height;
#pragma omp parallel for 
		for (int i=0; i<size; i++) buffer[i] = T(val);
		return *this;
	}
	template<typename _Tp>
	ImageT& SetImage(const ImageT<_Tp>& img, int _x, int _y){
		for (int i=0; i<img.height; i ++){
			for (int j=0; j<img.width; j ++){
				int x = j + _x;
				int y = i + _y;
				if (IsValid(x, y)){
					(*this)(x, y) = img(j, i);
				}
			}
		}
		return *this;
	}
	~ImageT(){
		if (buffer) buffer = NULL;
		if (roi)    delete roi;
	}
	ImageT& MallocBuffer(int w, int h){
		if (width == w && height == h){
			VERIFY_TRUE(buffer);
		}else{
			if (buffer) buffer = NULL;
			width = w;
			height = h;
			mem.resize(w*(h+1));
			buffer = &mem[0];
			//buffer = new T[w*h];
		}
		if (roi){
			delete roi;
			roi = NULL;
		}
		return *this;
	}
	FrameT<T> GetFrame() const{
		FrameT<T> frame(this->buffer, width, height, 0, this->roi);
		return frame;
	}
	ImageT& Transpose(){
		for (int i = 0, idx = 0; i < height; i++){
			for (int j = 0; j < width; j++, idx ++){
				T& v0 = buffer[idx];
				T& v1 = buffer[j*height+i];
				std::swap(v0, v1);
			}
		}
		std::swap(this->width, this->height);
		return *this;
	}
	ImageT& operator*=(T s){
		int size = width*height;
#pragma omp parallel for
		for (int i=0; i<size; i++){
			buffer[i] *= s;
		}
		return *this;
	}
	ImageT& operator*=(const ImageT& img){
		int size = width*height;
#pragma omp parallel for
		for (int i=0; i<size; i++){
			buffer[i] *= img.buffer[i];
		}
		return *this;
	}
	ImageT& operator/=(T s){
		if (IsZero(s)) this->Clear();
		else (*this) *= (1.0/s);	
		return *this;
	}
	ImageT& operator+=(T s){
		int size = width*height;
#pragma omp parallel for
		for (int i=0; i<size; i++){
			buffer[i] += s;
		}
		return *this;
	}
	ImageT& operator-=(T s){
		return (*this) += -s;
	}
	inline T& operator()(int x, int y) const{
		assert(IsValid(x, y));
		return buffer[y*width+x];
	}
	inline T operator()(float x, float y) const{
		assert(IsValid(x, y));
		int ax(x), ay(y);
		int bx(ax+1), by(ay+1);
		float cx(x-ax), cy(y-ay);
		float _cx(1-cx), _cy(1-cy);
		const T* pData0 = buffer + ay*width + ax;
		const T* pData1 = pData0 + width;
		T data0[2] = {0, 0};
		T data1[2] = {0, 0};
		if (IsValid(ax, ay))    data0[0] = pData0[0];
		if (IsValid(ax+1, ay))   data0[1] = pData0[1];
		if (IsValid(ax, ay+1)) data1[0] = pData1[0];
		if (IsValid(ax+1, ay+1)) data1[1] = pData1[1];

#if 1
		T val = (data0[0]*(_cx) + data0[1]*cx)*_cy + (data1[0]*(_cx) + data1[1]*cx)*cy;
#else
#pragma message("using sse bi-linear")
		__m128 mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7;
		mm0 = _mm_set_ps(cx,       _cx,        cx,       _cx);
		mm1 = _mm_set_ps(pData1[1], pData1[0], pData0[1], pData0[0]);
		mm2 = _mm_mul_ps(mm0, mm1);
		mm3 = _mm_shuffle_ps(mm2, mm2, _MM_SHUFFLE(2, 3, 0, 1));
		mm4 = _mm_add_ps(mm2, mm3);
		mm5 = _mm_set_ps(cy, cy, _cy, _cy);
		mm6 = _mm_mul_ps(mm4, mm5);
		mm7 = _mm_shuffle_ps(mm6, mm6, _MM_SHUFFLE(0, 1, 2, 3));
		mm0 = _mm_add_ps(mm6, mm7);
		T val = mm6.m128_f32[0] + mm6.m128_f32[2];
#endif
		return val;
	}
	inline T operator()(double x, double y) const{
		return (*this)(float(x), float(y));
	}
	inline bool IsValid(int x, int y) const{
		bool bValid = x>=0 && x<width && y>=0 && y<height;
		if (roi && bValid){
			bValid = x>=roi->x && x<=( roi->x + roi->width - 1) && y>=roi->y && y<=(roi->y+roi->height - 1);
		}
		return bValid;
	}
	inline bool IsValid(float x, float y) const{
		int ax(x), ay(y);
		//int bx(ax+1), by(ay+1);
		return IsValid(ax, ay)/* && IsValid(bx, by)*/;
	}
	inline bool IsValid(double x, double y) const{
		return IsValid(float(x), float(y));
	}
	inline T interp(float x, float y) const{
		if (IsValid(x, y)) return (*this)(x, y);
		return 0;
	}
	cv::Mat display(std::string name = "", const Rect32i* roi=NULL) const
	{
		return ::ShowImage<T>(name, buffer, width, height, T(0), T(0), roi);
	}
	void SetRoi(Rect32i rect){
		if (roi) delete roi;
		roi = NULL;
		if (rect.width == 0 && rect.height == 0){
		}else{
			roi = new Rect32i(rect);
		}
	}
	void Abs(){
		int size = width*height;
		for (int i=0; i<size; i++) buffer[i] = std::abs(buffer[i]);	
	}
	void Clear(){
		if (buffer && width > 0 && height > 0) memset(buffer, 0, sizeof(*buffer)*width*height);
	}
	void SetOutside(Rect32i rect, T val){
		for (int i=0; i<height; i ++){
			for (int j=0; j<width; j ++){
				if (j>=rect.x && j<= rect.x+rect.width-1 && i >= rect.y && i <= rect.y+rect.height-1){
				}else{
					(*this)(j, i) = val;
				}
			}
		}
	}
	ImageT SubImage(Rect32i rect, T val){
		ImageT img(rect.width, rect.height);
		img.SetValue(val);
		for (int i=0; i<img.height; i ++){
			for (int j=0; j<img.width; j ++){
				int x = j + rect.x;
				int y = i + rect.y;
				if (IsValid(x, y)){
					img(j, i) = (*this)(x, y);
				}
			}
		}
		return img;
	}
	bool LoadImageFile(std::string path){
#ifdef _CVLIB
		cv::Mat mt = cv::imread(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		ImageT<unsigned char> imSrc(mt.ptr(), mt.cols, mt.rows);
		this->MallocBuffer(imSrc.width, imSrc.height);
		int sz = width*height;
		Transform(imSrc.buffer, this->buffer, sz);
		return sz>0?true:false;
#else
		return false;
#endif
	}
	template<typename _Tp>
	bool LoadRawFile(std::string path, int w, int h){
		bool bRtn = false;
		bRtn = MallocBuffer(w, h).IsValid(0, 0);
		if (bRtn){
			std::vector<_Tp> vec(w*h+1);
			bRtn = LoadImageFromFile<_Tp>(path.c_str(), &vec[0], w, h);
			Transform(&vec[0], this->buffer, w*h);
		}
		return bRtn;
	}
	bool Save(std::string path){
		return WriteToFile<T, T>(path, buffer, width*height);
	}
	bool SaveCSV(std::string path){
		bool bRtn = false;
		FILE* fp = fopen(path.c_str(), "wt");
		if (fp){
			for (int i=0; i<width; i ++){
				fprintf(fp, "\"%d\"", i);
				if (i != width -1) fprintf(fp, ", ");
				else fprintf(fp, "\n");
			}
			for (int j=0; j<height; j ++){
				for (int i=0; i<width; i ++){
					fprintf(fp, "%f", (*this)(i, j));
					if (i != width -1) fprintf(fp, ", ");
					else fprintf(fp, "\n");
				}
			}
			fclose(fp);
			bRtn = true;
		}
		return bRtn;
	}
	ImageT& Dilate(int radius = 1){
		ImageT img = *this;
		//int radius = 1;
#ifndef _DEBUG
#pragma omp parallel for
#endif
		for (int y=0; y<height; y ++){
			for (int x=0; x<width; x ++){
				int _x, _y;
				bool bOut = false;
				if (img(x, y) == 0){
					for (int j=-radius; j<=radius&& !bOut; j ++){
						for (int i=-radius; i<=radius&& !bOut; i ++){
							_x = x + i;
							_y = y + j;
							if (!IsValid(_x, _y)) continue;
							if ((*this)(_x, _y) != 0){
								img(x, y) = 1;
								bOut = true;
							}
						}
					}
				}
			}
		}	

		*this = img;
		return *this;
	}
	ImageT& Reverse(){
		int size = width*height;
		for (int i=0; i<size; i ++) buffer[i] = buffer[i] == 0?1:0;
		return *this;
	}
	Rect32i* roi;
	T* buffer;
	int64 width, height;
private:
	std::vector<T> mem;
};

template<typename T>
ImageT<T> operator-(const ImageT<T>& imgL, const ImageT<T>& imgR)
{
	assert(imgL.width == imgR.width && imgL.height == imgR.height);
	int width = imgL.width;
	int height = imgL.height;
	ImageT<T> img(width, height);
	if (imgL.width == imgR.width && imgL.height == imgR.height){	
		for (int i=0; i<width*height; i++)
			img.buffer[i] = imgL.buffer[i] - imgR.buffer[i];
	}
	return img;
}

template<typename T> inline
	T& PointValue(T* data, int nx, int ny, int nz, int x, int y, int z){
		assert(nx>0 && ny > 0 && nz > 0 && x>=0 && y>= 0 && z >= 0 && x<nx && y<ny && z < nz);
		return (data + nx*ny*z)[y*nx+x];
}

template<typename T> inline
	T& PointValue(T* data, int nx, int ny, int x, int y){
		assert(nx > 0 && x >= 0 && x <= nx -1);
		assert(ny > 0 && y >= 0 && y <= ny -1);
		return (data + nx*y)[x];
}

template<typename T> inline
	T& PointValue(T* data, int nx, int x){
		assert(nx > 0 && x >= 0 && x <= nx -1);
		return data[x];
}

#define PtValue1D(data, width, x)                      ( data                          )[x              ]
#define PtValue2D(data, width, height, x, y)           ( (data) + (width)*(y)          )[x              ]
#define PtValue3D(data, width, height, depth, x, y, z) ( (data) + (width)*(height)*(z) )[(y)*(width)+(x)]


struct DrawGraphic{
	enum PAINT_TYPE{
		PAINT_ADD_VALUE,
		PAINT_SET_VALUE,
	};
	template<typename T>
	static void Circle(T* buffer, int width, int height, float centerX, float centerY, float radius, T value, PAINT_TYPE type = PAINT_ADD_VALUE){
		for (int i=0; i<height; i++){
			for (int j=0; j<width; j ++){
				float dx = j - centerX;
				float dy = i - centerY;
				float dis = sqrt(dx*dx + dy*dy) + 0.5;
				if (dis <= radius){
					if (type == PAINT_ADD_VALUE) buffer[width*i+j] += value;
					else if (type == PAINT_SET_VALUE) buffer[width*i+j] = value;
				}
			}
		}
	}
	template<typename T>
	static void Rectangle(T* buffer, int width, int height, Rect32i rc, T value, PAINT_TYPE type = PAINT_ADD_VALUE){
		int x0 = MAX(0, rc.x);
		int y0 = MAX(0, rc.y);
		int x1 = MIN(rc.width + rc.x, width-1);
		int y1 = MIN(rc.height + rc.y, height-1);
		for (int i=y0; i<=y1; i++){
			for (int j=x0; j<=x1; j++){
				if (type == PAINT_ADD_VALUE)  buffer[i*width + j] += value;
				else if (type == PAINT_SET_VALUE) buffer[i*width + j] = value;
			}
		}
	}
	template<typename T>
	static void Sphere(T* buffer, int nx, int ny, int nz, float centerX, float centerY, float centerZ, float radius, T value, PAINT_TYPE type = PAINT_ADD_VALUE){
#pragma omp parallel for
		for (int i=0; i<nz; i ++){
			for (int j=0; j<ny; j ++){
				for (int k=0; k<nx; k ++){
					float dx = k - centerX;
					float dy = j - centerY;
					float dz = i - centerZ;
					float dis = sqrt(dx*dx + dy*dy + dz*dz);
					if (dis <= radius){ 
						if (type == PAINT_ADD_VALUE)  PointValue<T>(buffer, nx, ny, nz, k, j, i) += value;
						else if (type == PAINT_SET_VALUE) PointValue<T>(buffer, nx, ny, nz, k, j, i) = value;
					}
				}
			}
		}
	}
};


template<typename T>
struct VolumeT
{
	typedef T DataType;
	VolumeT(std::string name = ""):buffer(NULL),DEF_VALUE(0),nx(0),ny(0),nz(0),strName(name){
	}
	template<typename _Tp>
	VolumeT(const _Tp* buf, int x, int y, int z):buffer(NULL),DEF_VALUE(0),nx(0),ny(0),nz(0){
		MallocBuffer(x, y, z);
		int64 size = nx*ny*nz;
		if (this->buffer && size > 0){
#pragma omp parallel for
			for (int64 i = 0; i < size; i++) this->buffer[i] = buf[i];
		}
	}
	VolumeT(int x, int y, int z):buffer(NULL),DEF_VALUE(0),nx(0),ny(0),nz(0){
		MallocBuffer(x, y, z);
		Zeros();
	}
	VolumeT(const VolumeT& vol):buffer(NULL),DEF_VALUE(0),nx(0),ny(0),nz(0){
		nx = ny = nz = 0;
		dx = dy = dz = 0;
		*this = vol;
	}
	virtual ~VolumeT(){
		FreeBuffer();
	}
	VolumeT& operator=(const VolumeT& vol) throw(){
		if (nx == vol.nx && ny == vol.ny && nz == vol.nz){
		}else{
			FreeBuffer();
			MallocBuffer(vol.nx, vol.ny, vol.nz);
		}
		if (buffer){
			memcpy(buffer, vol.buffer, sizeof(*buffer)*nx*ny*nz);
			DEF_VALUE = vol.DEF_VALUE;
			SetUint(vol.dx, vol.dy, vol.dz);
		}
		return *this;
	}
	VolumeT& MallocBuffer(int64 x, int64 y, int64 z) throw() {
		if (nx == x && ny == y && nz == z){
		}else if (x>0 && y>0 && z>0){
#ifdef _USE_VECTOR
			mem.resize(x*y*z+1);
			buffer = &mem[0];
#else
			buffer = new T[x*y*z + MAX3(x*y, y*z, z*x)];
#endif
			assert(buffer);
			nx = x;
			ny = y;
			nz = z;
		}
		dx = dy = dz = 0;
		Zeros();
		return *this;
	}
	void FreeBuffer() throw(){
#ifdef  _USE_VECTOR
		mem.resize(0);
		std::vector<T>().swap(mem);
		mem.clear();
#else
		if (buffer) delete []buffer;
#endif
		buffer = NULL;
		nx = ny = nz = 0;
		dx = dy = dz = 0;
	}
	VolumeT& Resize(double scale) throw(){
		int w = nx*scale + 0.5;
		int h = ny*scale + 0.5;
		int d = nz*scale + 0.5;
		return this->Resize(w, h, d);
	}	
	VolumeT& ResizeHalf(){
		VolumeT vol(nx/2, ny/2, nz/2);
		const int scale = 2;
#pragma omp parallel for
		for (int k=0; k<vol.nz; k ++){
			for (int j=0; j<vol.ny; j ++){
				for (int i=0; i<vol.nx; i ++){
					T sum = 0;
					T cnt = 0;
					for (int kk=0; kk<2; kk ++){
						for (int jj =0; jj<2; jj ++){
							for (int ii=0; ii<2; ii ++){
								int x = ii + i*scale;
								int y = jj + j*scale;
								int z = kk + k*scale;
								if (this->IsValid(x, y, z)){
									sum += (*this)(x, y, z);
									cnt ++;
								}
							}
						}
					}
					if (cnt > 0){
						vol(i, j, k) = sum/cnt;
					}else{
						VERIFY_TRUE(0);
					}
				}
			}
		}
		*this = vol;
		return *this;
	}
	VolumeT& Resize(int cx, int cy, int cz) throw(){
		std::cout<<"Resize("<<nx<<" "<<ny<<" "<<nz<<")->("<<cx<<" "<<cy<<" "<<cz<<")"<<std::endl;
		if (cx == nx && cy == ny && cz == nz){
			std::cout<<"Resize do nothing"<<std::endl;
		}else{
			if (cx == nx/2 && cy == ny/2 && cz == ny/2){
				return this->ResizeHalf();
			}else if (cx == nx/4 && cy == ny/4 && cz == ny/4){
				return ResizeHalf().ResizeHalf();
			}else if (cx == nx/8 && cy == ny/8 && cz == ny/8){
				return ResizeHalf().ResizeHalf().ResizeHalf();
			}else if (cx == nx/16 && cy == ny/16 && cz == ny/16){
				return ResizeHalf().ResizeHalf().ResizeHalf().ResizeHalf();
			}else{
				if (cx == nx && cy == ny && cz == nz){
				}else{
					VolumeT vol(cx, cy, cz);
					double sx = nx/double(cx);
					double sy = ny/double(cy);
					double sz = nz/double(cz);
					//VERIFY_TRUE(sx == sy && sx == sz);
					float half_sx = 0;
					float half_sy = 0;
					float half_sz = 0;
					if (sx > 1){
						half_sx = 0.5*(sx-1);
						half_sy = 0.5*(sy-1);
						half_sz = 0.5*(sz-1);
					}

					#pragma omp parallel for
					for (int k=0; k<cz; k ++){
						for (int j=0; j<cy; j ++){
							for (int i=0; i<cx; i ++){
								float _x = sx*i + half_sx;
								float _y = sy*j + half_sy;
								float _z = sz*k + half_sz;
								vol(i, j, k) = (*this)(_x, _y, _z);
							}
						}
					}
					*this = vol;
				}
			}
		}
		return *this;
	}

	Point3dT<T> GetCenter() const{
		return Point3dT<T>( (nx-1)/2.0, (ny-1)/2.0, (nz-1)/2.0 );
	}

	Size3dT<int> GetSize3D() const{
		return Size3dT<int>(nx, ny, nz);
	}
	VolumeT& Rotate(const Matrix2dReal4x4& mat) throw(){
		VolumeT vol(*this);

		T cx = (nx - 1) /2.0;
		T cy = (ny - 1) /2.0;
		T cz = (nz - 1) /2.0;

		//transpose
		Matrix2dReal4x4 mat4(
			mat.data[0], mat.data[4], mat.data[8], 0,
			mat.data[1], mat.data[5], mat.data[9], 0,
			mat.data[2], mat.data[6], mat.data[10], 0,
			0          ,           0,           0, 1);
		Matrix2dReal4x4_SSE mat_sse(mat4);
		#pragma omp parallel for
		for (int k=0; k<nz; k ++){
			for (int j=0; j<ny; j ++){
				for (int i=0; i<nx; i ++){
					Matrix2dReal1x4 dst(i - cx, cy - j, cz - k, 1), src;
					mat_sse.Multiple(dst, src);
					float x = src.data[0] + cx;
					float y = cy - src.data[1];
					float z = cz - src.data[2];
					vol(i, j, k) = (*this)(x, y, z);
				}
			}
		}
		*this = vol;
		return *this;
	}

	VolumeT& Rotate(const Matrix2dReal3x3& mat) throw(){
		Matrix2dReal4x4 mat4(
			mat.data[0], mat.data[1], mat.data[2], 0,
			mat.data[3], mat.data[4], mat.data[5], 0,
			mat.data[6], mat.data[7], mat.data[8], 0,
			0          ,           0,           0, 1);
		Rotate(mat4);
		return *this;
	}
	//rad unit
	VolumeT& Rotate(float angleX, float angleY, float angleZ) throw(){
		Rotate(MakeRotationMatrix(angleX, angleY, angleZ));
		return *this;
	}
	inline T& operator()(int x, int y, int z) const throw(){
		assert(IsValid(x, y, z));
		return (buffer + nx*ny*z)[y*nx+x];	
	}
	template<typename _Tp>
	inline bool IsValid(_Tp x, _Tp y, _Tp z) const throw(){
		return (x>=0 && x<=nx-1 && y>=0 && y<=ny-1 && z>= 0 && z<=nz-1);
	}
	inline T operator()(double x, double y, double z) const throw(){
		if (!IsValid(x, y, z)) return DEF_VALUE;
		return (*this)(float(x), float(y), float(z));
	}
	inline T operator()(float x, float y, float z) const throw(){
		//assert(IsValid(x, y, z)); 
		//if (!IsValid(x, y, z)) return DEF_VALUE;
		const int ax(x),      ay(y),    az(z);
		const float cx(x-ax), cy(y-ay), cz(z-az);
		const float _cx(1-cx), _cy(1-cy), _cz(1-cz);
		const int offsetXY0 = nx*ny*az + nx*ay + ax;
		const int offsetXY1 = offsetXY0 + nx*ny;
		const T* pDataA = buffer + offsetXY0;
		const T* pDataB = pDataA + nx*ny;
#if 1
		T tmp[8] = {DEF_VALUE, DEF_VALUE, DEF_VALUE, DEF_VALUE, DEF_VALUE, DEF_VALUE, DEF_VALUE, DEF_VALUE};

		for (int k=0; k<2; k ++){
			for (int j=0; j<2; j ++){
				for (int i=0; i<2; i ++){
					int _x = ax + i;
					int _y = ay + j;
					int _z = az + k;
					if (IsValid(_x, _y, _z)){
						PointValue(tmp, 2, 2, 2, i, j, k) = (*this)(_x, _y, _z);
					}
				}
			}
		}
		pDataA = tmp;
		pDataB = tmp + 4;
		float v0 = (pDataA[0]*(1-cx) + pDataA[1]*cx)*(1-cy) +  (pDataA[2]*(1-cx) + pDataA[2+1]*cx)*cy;
		float v1 = (pDataB[0]*(1-cx) + pDataB[1]*cx)*(1-cy) +  (pDataB[2]*(1-cx) + pDataB[2+1]*cx)*cy;
		T val = v0*(1-cz) + v1*cz;
		//float v0 = (pDataA[0]*(1-cx) + pDataA[1]*cx)*(1-cy) +  (pDataA[nx]*(1-cx) + pDataA[nx+1]*cx)*cy;
		//float v1 = (pDataB[0]*(1-cx) + pDataB[1]*cx)*(1-cy) +  (pDataB[nx]*(1-cx) + pDataB[nx+1]*cx)*cy;
		//T val = v0*(1-cz) + v1*cz;
#else
#pragma message("using sse tri-linear")
		__m128 mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7;
		mm0 = _mm_set_ps(cx,       _cx,        cx,       _cx);
		mm1 = _mm_set_ps(pDataA[nx+1], pDataA[nx], pDataA[1], pDataA[0]);
		mm2 = _mm_mul_ps(mm0, mm1);
		mm3 = _mm_shuffle_ps(mm2, mm2, _MM_SHUFFLE(2, 3, 0, 1));
		mm4 = _mm_add_ps(mm2, mm3);
		mm5 = _mm_set_ps(cy, cy, _cy, _cy);
		mm6 = _mm_mul_ps(mm4, mm5);

		mm1 = _mm_set_ps(pDataB[nx+1], pDataB[nx], pDataB[1], pDataB[0]);
		mm2 = _mm_mul_ps(mm0, mm1);
		mm3 = _mm_shuffle_ps(mm2, mm2, _MM_SHUFFLE(2, 3, 0, 1));
		mm4 = _mm_add_ps(mm2, mm3);
		mm7 = _mm_mul_ps(mm4, mm5);

		T val = (mm6.m128_f32[0] + mm6.m128_f32[2])*_cz +  (mm7.m128_f32[0] + mm7.m128_f32[2])*cz;
#endif

		return val;
	}
	template<typename D>
	VolumeT& operator*=(VolumeT<D>& vol) throw(){
		assert (vol.nx == nx && vol.ny == ny && vol.nz == nz);
		if (vol.nx == nx && vol.ny == ny && vol.nz == nz){
			int size = nx*ny*nz;
#pragma omp parallel for
			for (int i=0; i<size; i++) buffer[i] *= vol.buffer[i];	
		}
		return *this;
	}
	VolumeT& operator*=(float scale) throw(){
		int size = nx*ny*nz;
		if (buffer){
#pragma omp parallel for
			for (int i=0; i<size; i++) buffer[i] *= scale;
		}
		return *this;
	}
	template<typename D>
	VolumeT& operator+=(const VolumeT<D>& vol) throw(){
		int size = nx*ny*nz;
		if (vol.nx == nx && vol.ny == ny && vol.nz == nz){
			//#pragma omp parallel for
			//for (int i=0; i<size; i++) buffer[i] += vol.buffer[i];
			::Add_Vec(buffer, vol.buffer, buffer, size);
		}
		return *this;
	}
	template<typename D>
	VolumeT& Shift(D val) throw(){
		int size = nx*ny*nz;
		if (buffer){
#pragma omp parallel for
			for (int i=0; i<size; i++) buffer[i] += val;
		}
		return *this;
	}
	template<typename D>
	VolumeT& operator-=(const VolumeT<D>& vol) throw(){
		int size = nx*ny*nz;
		if (vol.nx == nx && vol.ny == ny && vol.nz == nz){
#pragma omp parallel for
			for (int i=0; i<size; i++) buffer[i] -= vol.buffer[i];
		}
		return *this;
	}
	template<typename D>
	VolumeT& operator/=(const VolumeT<D>& vol) throw(){
		assert (vol.nx == nx && vol.ny == ny && vol.nz == nz);
		if (vol.nx == nx && vol.ny == ny && vol.nz == nz){
			int size = nx*ny*nz;
#pragma omp parallel for
			for (int i=0; i<size; i++){
				if (!IsZero(vol.buffer[i])){
					buffer[i] /= vol.buffer[i];
					//if (_isnan(buffer[i])) buffer[i] = 0;
				}else buffer[i] = 0;
			}
		}
		return *this;
	}	
	VolumeT& operator/=(float v) throw(){
		int size = nx*ny*nz;
		if (IsZero(v)) memset(buffer, 0, sizeof(buffer[0])*size);
		else{
			(*this) *= float(1.0)/v;
		}
		return *this;
	}	
	VolumeT& log() throw(){
		int size = nx*ny*nz;
#pragma omp parallel for
		for (int i=0; i<size; i++) buffer[i] = std::log(double(buffer[i]));
		return *this;
	}
	VolumeT& abs() throw(){
		int size = nx*ny*nz;
#pragma omp parallel for
		for (int i=0; i<size; i++) buffer[i] = std::abs(buffer[i]);
		return *this;
	}
	VolumeT& exp() throw(){
		int size = nx*ny*nz;
#pragma omp parallel for
		for (int i=0; i<size; i++) buffer[i] = std::exp(double(buffer[i]));
		return *this;
	}	
	T* GetDataZ(int z) const throw(){
		if (z>=0 && z < nz) return (buffer + nx*ny*z);
		return NULL;
	}
	VolumeT& Zeros() throw(){
		if (buffer) memset(buffer, 0, sizeof(*buffer)*nx*ny*nz);
		return *this;
	}
	VolumeT& Ones() throw(){
		SetValue(1);	
		return *this;
	}
	VolumeT& SetValue(T val) throw(){
		int size = nx*ny*nz;
#pragma omp parallel for
		for (int i=0; i<size; i++) buffer[i] = val;
		return *this;
	}
	VolumeT& ClearOutside(Box32i box, T val) throw(){
#pragma omp parallel for
		for (int i=0; i<nz; i ++){
			for (int j=0; j<ny; j ++){
				for (int k=0; k<nx; k ++){
					if (k>=box.x && k<=box.x+box.width-1
						&& j>=box.y && j <= box.y+box.height-1
						&& i>=box.z && i<=box.z+box.depth-1)
					{
					}else{
						(*this)(k, j, i) = val;
					}
				}
			}
		}
		return *this;
	}
	VolumeT SubVolume(Box32i box, T val) throw(){
		VolumeT vol(box.width, box.height, box.depth);
//#pragma omp parallel for
		for (int i=0; i<box.depth; i ++){
			for (int j=0; j<box.height; j ++){
				for (int k=0; k<box.width; k ++){
					Point3d32i pt(k+box.x, j+box.y, i+box.z);
					if (IsValid(pt.x, pt.y, pt.z)){
						vol(k, j, i) = (*this)(pt.x, pt.y, pt.z);
					}else{
						vol(k, j, i) = val;
					}
				}
			}
		}
		return vol;
	}
	VolumeT& SetVolume(const VolumeT& vol, Point3d32i point) throw(){
#pragma omp parallel for
		for (int i=0; i<vol.nz; i ++){
			for (int j=0; j<vol.ny; j ++){
				for (int k=0; k<vol.nx; k ++){
					Point3d32i pt(k+point.x, j+point.y, i+point.z);
					if (IsValid(pt.x, pt.y, pt.z)){
						(*this)(pt.x, pt.y, pt.z) = vol(k, j, i);
					}
				}
			}
		}
		return *this;
	}
	VolumeT& SetVolume(const VolumeT& vol, Box32i boxSrc, Point3d32i ptDst) throw(){
#pragma omp parallel for
		for (int i=0; i<boxSrc.depth; i ++){
			for (int j=0; j<boxSrc.height; j ++){
				for (int k=0; k<boxSrc.width; k ++){
					Point3d32i ptSrc(k+boxSrc.x, j+boxSrc.y, i+boxSrc.z);
					if (vol.IsValid(ptSrc.x, ptSrc.y, ptSrc.z)){
						Point3d32i pt(ptDst.x + k, ptDst.y + j, ptDst.z + i);
						if (this->IsValid(pt.x, pt.y, pt.z)){
							(*this)(pt.x, pt.y, pt.z) = vol(ptSrc.x, ptSrc.y, ptSrc.z);
						}
					}
				}
			}
		}
		return *this;
	}
	void GeneratorShape1(){
		std::vector<T> tmp(nx*ny+1);
		T* pBuf = &tmp[0];
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx/3, ny/3, (nx+ny)/16.0*1.2, 1);
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx/3+nx/32, ny/3, (nx+ny)/16.0*0.4, 0);
		Rect32i rc(nx/2 , ny/2 , nx/2-nx/5, ny/2-ny/4);
		DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, 1);
		DrawGraphic::Circle<T>(pBuf, nx, ny, rc.x+rc.width/2, rc.y+rc.height/2, (nx+ny)/16.0*0.4, 0);
#pragma omp parallel for
		for (int i=0; i<nz; i++){
			memcpy(GetDataZ(i), pBuf, sizeof(T)*nx*ny);
		}
	}
	void GeneratorShape2(){
		this->Zeros();
		DrawGraphic::Sphere<T>(buffer, nx, ny, nz, nx*2/5.0, ny*2/5.0, nz*2/5.0, nx/3, 1);
	}
	void GeneratorShape3(){
		std::vector<T> tmp(nx*ny+1);
		T* pBuf = &tmp[0];
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx/3, ny/3, (nx+ny)/16.0*1.2, 1);
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx*2/3, ny/3, (nx+ny)/16.0*1, 1);
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx/3, ny*2/3, (nx+ny)/16.0*0.6, 1);
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx/3+nx/32, ny/3, (nx+ny)/16.0*0.4, 0);
		Rect32i rc(nx/2 , ny/2 , nx/2-nx/5, ny/2-ny/4);
		DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, 1);
		DrawGraphic::Circle<T>(pBuf, nx, ny, rc.x+rc.width/2, rc.y+rc.height/2, (nx+ny)/16.0*0.4, 0);
#pragma omp parallel for
		for (int i=0; i<nz; i++){
			memcpy(GetDataZ(i), pBuf, sizeof(T)*nx*ny);
		}
	}
	void GeneratorShape4(){
		std::vector<T> tmp(nx*ny+1);
		T* pBuf = &tmp[0];
		int cx = 72;
		int cy = 255;
		int w = 4;
		int h = 8;
		int value = 7;
		{
			Rect32i rc(cx-w/2 , cy-h/2 , w, h);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, value--);
		}
		{
			w *= 2; h *= 2;
			cx += w + 4;
			Rect32i rc(cx-w/2 , cy-h/2 , w, h);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, value--);
		}
		{
			w *= 1.6; h *= 2;
			cx += w + 4;
			Rect32i rc(cx-w/2 , cy-h/2 , w, h);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, value--);
		}
		{
			w *= 1.6; h *= 2;
			cx += w + 4;
			Rect32i rc(cx-w/2 , cy-h/2 , w, h);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, value--);
		}
		{
			w *= 1.6; h *= 2;
			cx += w + 4;
			Rect32i rc(cx-w/2 , cy-h/2 , w, h);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, value--);
		}
		{
			w *= 1.5; h *= 1.5;
			cx += w + 4;
			Rect32i rc(cx-w/2 , cy-h/2 , w, h);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, value--);
		}
		{
			w *= 2.2;
			h *= 1.5;
			cx += w+4;
			Rect32i rc(cx-w/2 , cy-h/2 , w, h);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, value--);

			cx = rc.x;
			cy = rc.y;
		}

		w = 64;
		h = 8;
		for (int i=0; i < 8; i ++){
			Rect32i rc(cx+16 , cy+h*(2*i+1) , w, h);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, value++);
		}

		//{
		//	w *= 1.4;
		//	h *= 0.8;
		//	cx += w+4;
		//	Rect32i rc(cx-w/2 , cy-h/2 , w, h);
		//	DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, value--);
		//}
#pragma omp parallel for
		for (int i=96; i<nz - 96; i++){
			memcpy(GetDataZ(i), pBuf, sizeof(T)*nx*ny);
		}
	}
	void GeneratorShape5(){
		DrawGraphic::Sphere<T>(buffer, nx, ny, nz, nx*0.5, ny*0.5, nz*0.5, nx/4, 1);
	}
	void GeneratorShape6(){
		T cx(nx*0.5 - 32), cy(ny*0.5 - 32), cz(nz*0.5);
		DrawGraphic::Sphere<T>(buffer, nx, ny, nz, cx, cy, cz, nx/4, 1);
		DrawGraphic::Sphere<T>(buffer, nx, ny, nz, cx, cy, cz, nx/4-12, -1);
		DrawGraphic::Sphere<T>(buffer, nx, ny, nz, cx+32, cy-32, cz, 24, 1);
	}
	void GeneratorShape7(){
		std::vector<T> tmp(nx*ny+1);
		T* pBuf = &tmp[0];
		int w = nx *4 / 7;
		int h = ny *4 / 9;
		int d = nz / 2;
		Rect32i rc(nx/2 - w /2, ny/2-h/2, w, h);
		DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, 1);
		for (int i=0; i<d; i ++){
			int idx = nz/2 - d/2 + i;
			memcpy(GetDataZ(idx), pBuf, nx*ny*sizeof(*pBuf));
		}
	}
	//cylinder
	void GeneratorShapeSim1(){
		std::vector<T> tmp(nx*ny+1);
		T* pBuf = &tmp[0];
		//Rect32i rc(nx/2-360/2, ny/2-360/2, 360, 360);
		//DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, 1);
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx/2, ny/2, 420/2, 1, DrawGraphic::PAINT_SET_VALUE);
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx/2-328.166/2, ny/2, 63.549/2, 2.7, DrawGraphic::PAINT_SET_VALUE);
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx/2+328.166/2, ny/2, 63.549/2, 2.7, DrawGraphic::PAINT_SET_VALUE);
		for (int i=0; i<6; i ++){
			int idx = i + nz/2 - 5.497/2;
			memcpy(GetDataZ(idx), pBuf, nx*ny*sizeof(*pBuf));
		}
	}
	//rectangle
	void GeneratorShapeSim2(){
		std::vector<T> tmp(nx*ny+1);
		T* pBuf = &tmp[0];
		//Rect32i rc(nx/2-360/2, ny/2-360/2, 360, 360);
		//DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, 1);
		DrawGraphic::Circle<T>(pBuf, nx, ny, nx/2, ny/2, 420/2, 1, DrawGraphic::PAINT_SET_VALUE);
		{
			Rect32i rc(nx/2-328.166/2 - 63.549/2, ny/2 - 63.549/2, 63.549, 63.549);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, 2.7, DrawGraphic::PAINT_SET_VALUE);
		}
		{
			Rect32i rc(nx/2+328.166/2 - 63.549/2, ny/2 - 63.549/2, 63.549, 63.549);
			DrawGraphic::Rectangle<T>(pBuf, nx, ny, rc, 2.7, DrawGraphic::PAINT_SET_VALUE);
		}
		for (int i=0; i<6; i ++){
			int idx = i + nz/2 - 5.497/2;
			memcpy(GetDataZ(idx), pBuf, nx*ny*sizeof(*pBuf));
		}
	}
	void ClearOutofRadiusData(float radius){
		std::vector<T> tmp(nx*ny+1);
		float cx = (nx-1)/2.0;
		float cy = (ny-1)/2.0;
		for (int i=0; i<nz; i++){
			T* img = GetDataZ(i);
			for (int y=0; y<ny; y++){
				for (int x=0; x<nx; x++){
					float dx = x - cx;
					float dy = y - cy;
					if (dx*dx + dy*dy < radius*radius){
						tmp[y*ny+x] = img[y*ny+x];
					}
				}
			}
			memcpy(img, &tmp[0], sizeof(T)*nx*ny);
		}
	}
	void SliceX(int x) const{
		char szName[256];
		sprintf(szName, "%s:slice X", strName.c_str());
		if (x>=0 && x <= nx-1 ){
			std::vector<T> tmp(nx*nz+16);
			T* buf = &tmp[0];
			#pragma omp parallel for
			for (int i=0; i<nz; i++){
				T* pData = GetDataZ(i) + x;
				for (int j=0; j<ny; j++){
					buf[i*ny+j] = pData[j*nx];
				}
			}
			ShowImage(std::string(szName),buf, nx, nz, T(0), T(0));
		}else{
			std::cout<<"warning no slice "<<szName<<"="<<x;
		}
	}
	void SliceY(int y) const{
		char szName[256];
		sprintf(szName, "%s:slice Y", strName.c_str());		
		if (y>=0 && y<=ny-1){
			std::vector<T> tmp(nx*nz+16);
			T* buf = &tmp[0];
			#pragma omp parallel for
			for (int i=0; i<nz; i++){
				T* pData = GetDataZ(i) + y*nx;
				for (int j=0; j<nx; j++){
					buf[i*nx+j] = pData[j];
				}
			}
			T vMax(0), vMin(0);
			ShowImage(std::string(szName),buf, nx, nz, vMax, vMin);
		}else{
			std::cout<<"warning no slice "<<szName<<"="<<y;
		}
	}
	void SliceZ(int z) const throw(){
		char szName[256];
		sprintf(szName, "%s:slice Z", strName.c_str());
		if (z >= 0 && z <= nz -1){
			//T vMax(2000), vMin(-1200);
			T vMax(0), vMin(0);
			//GetMaxMin(this->buffer, nx*ny*nz, vMax, vMin);
			//auto min_max = std::minmax(buffer, buffer+nx*ny*nz);
			ShowImage(std::string(szName),this->GetDataZ(z), nx, ny, vMax, vMin);
		}else{
			::ClearCurrentLine();
			std::cout<<"warning no slice "<<szName<<"="<<z;
		}
	}
	VolumeT& Dilate(int radius = 1) throw(){
		std::cout<<"Volume Dilate="<<radius<<std::endl;
		VolumeT vol = *this;
		//int radius = 1;
#pragma omp parallel for
		for (int z=0; z<nz; z ++){
			for (int y=0; y<ny; y ++){
				for (int x=0; x<nx; x ++){
					int _x, _y, _z;
					bool bOut = false;
					for (int k=-radius; k<=radius && !bOut; k ++){
						for (int j=-radius; j<=radius&& !bOut; j ++){
							for (int i=-radius; i<=radius&& !bOut; i ++){
								_x = x + i;
								_y = y + j;
								_z = z + k;
								if (!IsValid(_x, _y, _z)) continue;
								if ((*this)(_x, _y, _z)){
									vol(x, y, z) = 1;
									bOut = true;
								}
							}
						}
					}
				}
			}	
		}
		*this = vol;
		return *this;
	}
	template<typename D>
	VolumeT& FilterXYZ(const std::vector<D>& kernel){
		return FilterZ<D>(kernel).FilterX<D>(kernel).FilterY<D>(kernel);
	}
	template<typename D>
	VolumeT& FilterX(const std::vector<D>& kernel)
	{
		std::cout<<"Volume begin FilterX ";

#pragma omp parallel for
		for (int z=0; z<nz; z ++){
			T* dataZ = this->GetDataZ(z);
			for (int y=0; y<ny; y ++){
				int offset = y*nx;
				int step = 1;
				FilterStep(kernel, dataZ + offset, dataZ + offset, nx, step, step); 
			}
		}

		std::cout<<", end FilterX"<<std::endl;
		return *this;
	}

	template<typename D>
	VolumeT& FilterY(const std::vector<D>& kernel) throw()
	{
		std::cout<<"Volume begin FilterY ";

#pragma omp parallel for
		for (int z=0; z<nz; z ++){
			T* dataZ = this->GetDataZ(z);
			for (int x=0; x<nx; x ++){
				int offset = x;
				int step = nx;
				FilterStep(kernel, dataZ + offset, dataZ + offset, ny, step, step); 
			}
		}

		std::cout<<", end FilterY"<<std::endl;
		return *this;
	}

	template<typename D>
	VolumeT& FilterZ(const std::vector<D>& kernel) throw()
	{
		std::cout<<"Volume begin FilterZ ";

#pragma omp parallel for
		for (int y=0; y<ny; y ++){
			for (int x=0; x<nx; x ++){
				int offset = y*nx + x;
				int step = nx*ny;
				FilterStep(kernel, buffer + offset, buffer + offset, nz, step, step);
			}
		}

		std::cout<<", end FilterZ"<<std::endl;
		return *this;
	}
	void FlipX() throw(){
		const int half_x = nx/2;
#pragma omp parallel for
		for (int x=0; x<half_x; x ++){
			for (int z=0; z<nz; z ++){
				for (int y=0; y<ny; y ++){
					T& A = PointValue(buffer, nx, ny, nz, x, y, z);
					T& B = PointValue(buffer, nx, ny, nz, nx-1-x, y, z);				
					std::swap(A, B);
				}
			}
		}
		printf("Flip X\n");
	}

	void FlipY() throw(){
		const int half_y = ny/2;
#pragma omp parallel for
		for (int y=0; y<half_y; y ++){
			for (int z=0; z<nz; z ++){
				for (int x=0; x<nx; x ++){
					T& A = PointValue(buffer, nx, ny, nz, x, y, z);
					T& B = PointValue(buffer, nx, ny, nz, x, ny-1-y, z);				
					std::swap(A, B);
				}
			}
		}
		printf("Flip Y\n");
	}

	void FlipZ() throw(){
		const int half_z = nz/2;
#pragma omp parallel for
		for (int z=0; z<half_z; z ++){
			for (int y=0; y<ny; y ++){
				for (int x=0; x<nx; x ++){
					T& A = PointValue(buffer, nx, ny, nz, x, y, z);
					T& B = PointValue(buffer, nx, ny, nz, x, y, nz-1-z);
					std::swap(A, B);
				}
			}
		}
		printf("Flip Z\n");
	}
	void SaveRawZ(std::string name, int z) throw(){
		::MakePathDirIfNeed(name.c_str());
		FILE* fp = fopen(name.c_str(), "wb");
		if (fp){
			fwrite(GetDataZ(z), sizeof(T), nx*ny, fp);
			fclose(fp);
		}
	}
	void SetName(std::string name) throw(){
		strName = name;
	}
	bool Save(std::string path) const throw(){
		bool bRtn = false;
		::MakePathDirIfNeed(path.c_str());
		FILE* fp = fopen(path.c_str(), "wb");
		if (fp){
			{int _nx = nx; fwrite(&_nx, sizeof(_nx), 1, fp); }
			{int _ny = ny; fwrite(&_ny, sizeof(_ny), 1, fp); }
			{int _nz = nz; fwrite(&_nz, sizeof(_nz), 1, fp); }
			T* p = buffer;
			for (int64 i = 0; i < nz; i++, p += nx*ny)
			{
				if (nx*ny == fwrite(p, sizeof(*p), nx*ny, fp)) {	 
					bRtn = true;
				}else{
					bRtn = false;
					break;
				}
			}
				
			fclose(fp);	
		}	
		printf("%s, Volume_Save:%s\n", bRtn?"successed":"failed", path.c_str());
		return bRtn;
	}
	template<typename D>
	bool LoadFolder(std::string path, int width, int height, int depth) throw(){
		std::cout<<"LoadFolder:"<<path<<"("<<width<<","<<height<<")";
		MallocBuffer(width, height, depth);
		std::vector<T*> vecDst(depth);
		for (int i=0; i<depth; i ++) vecDst[i] = this->buffer + width*height*i;
		bool bRtn = LoadProjectionImageFolder<D>(path.c_str(), width, height, vecDst);
		std::cout<<(bRtn?"successed":"failed")<<std::endl;
		return bRtn;
	}
	bool Load(std::string path){
		bool bRtn = false;
		FILE* fp = fopen(path.c_str(), "rb");
		if (fp){
			int nRead = 0;
			int _nx, _ny, _nz;
			nRead += fread(&_nx, 4, 1, fp);
			nRead += fread(&_ny, 4, 1, fp);
			nRead += fread(&_nz, 4, 1, fp);
			if (nRead == 3){
				const int64 size = int64(_nx)*int64(_ny)*int64(_nz) ;
				if (size > 0){
					MallocBuffer(_nx, _ny, _nz);
					VERIFY_TRUE (buffer);
					if (buffer){
						nRead = fread(buffer, sizeof(*buffer), size, fp);
						if (nRead == size) bRtn = true;
					}
				}
			}
			fclose(fp);
		}
		printf("%s, Volue_Load:%s\n", bRtn?"successed":"failed", path.c_str());
		return bRtn;
	}
	int64 BufferSize() const throw() {
		return int64(nx)*int64(ny)*int64(nz);
	}
	T* Buffer() const throw(){
		return buffer;
	}
	void SetUint(float _dx, float _dy, float _dz){
		dx = _dx; dy = _dy; dz = _dz;
	}
	Size3d32i Size() const throw(){
		return Size3d32i(nx, ny, nz);
	}
	T SetDefaultValue(T val){
		DEF_VALUE = val;
		return DEF_VALUE;
	}
#ifdef _USE_VECTOR
	std::vector<T> mem;
#endif
	T DEF_VALUE;
	T* buffer;
	int nx, ny, nz;
	float dx, dy, dz;
	std::string strName;
};

template<typename T> inline
	VolumeT<T> operator+(const VolumeT<T>& lVol, const VolumeT<T>& rVol)
{
	assert(lVol.nx == rVol.nx && lVol.ny == lVol.ny && lVol.nz == rVol.nz);
	VolumeT<T> vol = lVol;
	vol += rVol;
	return vol;
}



typedef VolumeT<float> Volume;


template<typename _TpVolumeA,typename _TpVolumeB, typename _TpVolumeC, typename _TpVolumeD, typename _TpVolumeE, typename _TpVolumeF, typename _TpVolumeG, typename _TpVolumeH, typename _TpVolumeI, typename _TpVolumeJ, typename _TpVolumeK, typename _TpVolumeL> inline
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC, const _TpVolumeD& volD, const _TpVolumeE& volE, const _TpVolumeF& volF, const _TpVolumeG& volG, const _TpVolumeH& volH, const _TpVolumeI& volI, const _TpVolumeJ& volJ, const _TpVolumeK& volK, const _TpVolumeL& volL) throw()
{
	std::cout<<"ShowVolume"<<std::endl;
	#ifdef _CVLIB
	Size3d32i volSize = volA.Size();
	std::vector<const void*> vecPtr;
	std::vector<char> vecIdx;
	vecPtr.push_back(&volA); vecIdx.push_back('a');
	if (std::find(vecPtr.begin(), vecPtr.end(), &volB) == vecPtr.end()){ vecPtr.push_back(&volB); vecIdx.push_back('b');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volC) == vecPtr.end()){ vecPtr.push_back(&volC); vecIdx.push_back('c');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volD) == vecPtr.end()){ vecPtr.push_back(&volD); vecIdx.push_back('d');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volE) == vecPtr.end()){ vecPtr.push_back(&volE); vecIdx.push_back('e');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volF) == vecPtr.end()){ vecPtr.push_back(&volF); vecIdx.push_back('f');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volG) == vecPtr.end()){ vecPtr.push_back(&volG); vecIdx.push_back('g');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volH) == vecPtr.end()){ vecPtr.push_back(&volH); vecIdx.push_back('h');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volI) == vecPtr.end()){ vecPtr.push_back(&volI); vecIdx.push_back('i');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volJ) == vecPtr.end()){ vecPtr.push_back(&volJ); vecIdx.push_back('j');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volK) == vecPtr.end()){ vecPtr.push_back(&volK); vecIdx.push_back('k');}
	if (std::find(vecPtr.begin(), vecPtr.end(), &volL) == vecPtr.end()){ vecPtr.push_back(&volL); vecIdx.push_back('l');}
	for (int i=0; ; i ++){
		ClearCurrentLine();
		std::cout<<"slice #"<<i;
		while (i>=volSize.cz) i-= volSize.cz;
		while (i < 0) i += volSize.cz;
		const int sz = vecIdx.size();
		for (int k=0; k<sz; k ++){
			const int idx = vecIdx[k];
			switch (vecIdx[k])
			{
			case 'a': volA.SliceZ(i); break;
			case 'b': volB.SliceZ(i); break;
			case 'c': volC.SliceZ(i); break;
			case 'd': volD.SliceZ(i); break;
			case 'e': volE.SliceZ(i); break;
			case 'f': volF.SliceZ(i); break;
			case 'g': volG.SliceZ(i); break;
			case 'h': volH.SliceZ(i); break;
			case 'i': volI.SliceZ(i); break;
			case 'j': volJ.SliceZ(i); break;
			case 'k': volK.SliceZ(i); break;
			case 'l': volL.SliceZ(i); break;
			default:
				break;
			}
		}
		int key = WaitKey();
		if (key == '-') i -= 2;
		if (key == 27) break;
	}
#endif
}

template<typename _TpVolumeA,typename _TpVolumeB, typename _TpVolumeC, typename _TpVolumeD, typename _TpVolumeE, typename _TpVolumeF, typename _TpVolumeG, typename _TpVolumeH, typename _TpVolumeI, typename _TpVolumeJ, typename _TpVolumeK> inline	
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC, const _TpVolumeD& volD, const _TpVolumeE& volE, const _TpVolumeF& volF, const _TpVolumeG& volG, const _TpVolumeH& volH, const _TpVolumeI& volI, const _TpVolumeJ& volJ, const _TpVolumeK& volK) throw()
{
	return ShowVolume(volA, volB, volC, volD, volE, volF, volG, volH, volI, volJ, volK, volK);
}

template<typename _TpVolumeA,typename _TpVolumeB, typename _TpVolumeC, typename _TpVolumeD, typename _TpVolumeE, typename _TpVolumeF, typename _TpVolumeG, typename _TpVolumeH, typename _TpVolumeI, typename _TpVolumeJ> inline	
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC, const _TpVolumeD& volD, const _TpVolumeE& volE, const _TpVolumeF& volF, const _TpVolumeG& volG, const _TpVolumeH& volH, const _TpVolumeI& volI, const _TpVolumeJ& volJ) throw()
{
	return ShowVolume(volA, volB, volC, volD, volE, volF, volG, volH, volI, volJ, volJ);
}

template<typename _TpVolumeA,typename _TpVolumeB, typename _TpVolumeC, typename _TpVolumeD, typename _TpVolumeE, typename _TpVolumeF, typename _TpVolumeG, typename _TpVolumeH, typename _TpVolumeI> inline	
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC, const _TpVolumeD& volD, const _TpVolumeE& volE, const _TpVolumeF& volF, const _TpVolumeG& volG, const _TpVolumeH& volH, const _TpVolumeI& volI) throw()
{
	return ShowVolume(volA, volB, volC, volD, volE, volF, volG, volH, volI, volI);
}

template<typename _TpVolumeA,typename _TpVolumeB, typename _TpVolumeC, typename _TpVolumeD, typename _TpVolumeE, typename _TpVolumeF, typename _TpVolumeG, typename _TpVolumeH> inline	
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC, const _TpVolumeD& volD, const _TpVolumeE& volE, const _TpVolumeF& volF, const _TpVolumeG& volG, const _TpVolumeH& volH) throw()
{
	return ShowVolume(volA, volB, volC, volD, volE, volF, volG, volH, volH);
}

template<typename _TpVolumeA, typename _TpVolumeB, typename _TpVolumeC, typename _TpVolumeD, typename _TpVolumeE, typename _TpVolumeF, typename _TpVolumeG> inline
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC, const _TpVolumeD& volD, const _TpVolumeE& volE, const _TpVolumeF& volF, const _TpVolumeG& volG) throw(){
	return ShowVolume(volA, volB, volC, volD, volE, volF, volG, volG);
}

template<typename _TpVolumeA, typename _TpVolumeB, typename _TpVolumeC, typename _TpVolumeD, typename _TpVolumeE, typename _TpVolumeF> inline
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC, const _TpVolumeD& volD, const _TpVolumeE& volE, const _TpVolumeF& volF) throw(){
	return ShowVolume(volA, volB, volC, volD, volE, volF, volF);
}

template<typename _TpVolumeA, typename _TpVolumeB, typename _TpVolumeC, typename _TpVolumeD, typename _TpVolumeE> inline
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC, const _TpVolumeD& volD, const _TpVolumeE& volE) throw(){
	return ShowVolume(volA, volB, volC, volD, volE, volE);
}

template<typename _TpVolumeA, typename _TpVolumeB, typename _TpVolumeC, typename _TpVolumeD> inline
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC, const _TpVolumeD& volD) throw(){
	return ShowVolume(volA, volB, volC, volD, volD);
}

template<typename _TpVolumeA, typename _TpVolumeB, typename _TpVolumeC> inline
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB, const _TpVolumeC& volC) throw(){
	return ShowVolume(volA, volB, volC, volC);
}

template<typename _TpVolumeA, typename _TpVolumeB> inline
	void ShowVolume(const _TpVolumeA& volA, const _TpVolumeB& volB) throw(){
	return ShowVolume(volA, volB, volB);
}

template<typename _TpVolume> inline
	void ShowVolume(const _TpVolume& vol) throw(){
	return ShowVolume(vol, vol);
}


template<typename T> inline
	bool Stitch(const VolumeT<T>& img1, Point3d32f offset1, const VolumeT<T>& img2, Point3d32f offset2, const VolumeT<T>& img3, Point3d32f offset3, const VolumeT<T>& img4, Point3d32f offset4, VolumeT<T>& img)
{
	const int IMG_COUNT = 4;
	const VolumeT<T>* pVolume[IMG_COUNT] = {&img1, &img2, &img3, &img4,};
	Box32f box1(0, 0, 0, img1.nx, img1.ny, img1.nz);
	Box32f box2(0, 0, 0, img2.nx, img2.ny, img2.nz);
	Box32f box3(0, 0, 0, img3.nx, img3.ny, img3.nz);
	Box32f box4(0, 0, 0, img4.nx, img4.ny, img4.nz);
	Box32f* pBox[IMG_COUNT] = {&box1, &box2, &box3, &box4};
	Point3d32f* pOffset[IMG_COUNT] = {&offset1, &offset2, &offset3, &offset4};
	Point3d32f ptMIN(box1.x, box1.y, box1.z);
	Point3d32f ptMAX(box1.x, box1.y, box1.z);
	for (int i=0; i<IMG_COUNT; i ++){
		pBox[i]->Offset(pOffset[i][0]);
		ptMIN.x = MIN(pBox[i]->x, ptMIN.x);
		ptMIN.y = MIN(pBox[i]->y, ptMIN.y);
		ptMIN.z = MIN(pBox[i]->z, ptMIN.z);
		ptMAX.x = MAX(pBox[i]->x + pBox[i]->width,  ptMAX.x);
		ptMAX.y = MAX(pBox[i]->y + pBox[i]->height, ptMAX.y);
		ptMAX.z = MAX(pBox[i]->z + pBox[i]->depth,  ptMAX.z);
	}
	Box32i boxRes(0, 0, 0, 0, 0, ptMAX.z - ptMIN.z + 0.5);
	boxRes.width = boxRes.height = MAX(ptMAX.x - ptMIN.x + 0.5, ptMAX.y - ptMIN.y + 0.5);
	img.MallocBuffer(boxRes.width, boxRes.height, boxRes.depth);
	const Point3d32i shift(-ptMIN.x, -ptMIN.y, -ptMIN.z);
	//VolumeT<char> sum(img.nx, img.ny, img.nz);
	img.SetValue(-1024);
	for (int idx=0; idx<IMG_COUNT; idx ++){
		const Size3d32i size(pBox[idx]->width + 0.5, pBox[idx]->height + 0.5, pBox[idx]->depth + 0.5);
		const Point3d32f& offset = pOffset[idx][0];
		const VolumeT<T>& vol = pVolume[idx][0];
		bool bExisted = false;
		for (int m=0; m<idx-1; m ++){
			if (pVolume[m] == pVolume[idx] && pOffset[m][0] == pOffset[idx][0]){
				bExisted = true;
				break;
			}
		}
		if (bExisted) continue;
		#pragma omp parallel for
		for (int k=0; k<size.cz; k ++){
			for (int i=0; i<size.cy; i ++){
				for (int j=0; j<size.cx; j ++){
					Point3d32i xyz(j+offset.x, i+offset.y, k+offset.z);
					xyz += shift;
					if (img.IsValid(xyz.x, xyz.y, xyz.z)){
						T& val = img(xyz.x, xyz.y, xyz.z);
						val = MAX(vol(j, i, k),val);
					}
				}
			}
		}
	}
	return true;
}

template<typename T> inline
	bool Stitch(const VolumeT<T>& imgRef, const VolumeT<T>& imgMove1, Point3d32f offset1, const VolumeT<T>& imgMove2, Point3d32f offset2, VolumeT<T>& img)
{
#if 1
	return Stitch(imgRef, Point3d32f(0, 0, 0), imgMove1, offset1, imgMove2, offset2, imgMove2, offset2, img);
#else

	Box32f boxRef(0, 0, 0, imgRef.nx, imgRef.ny, imgRef.nz);
	Box32f boxMove1(0, 0, 0, imgMove1.nx, imgMove1.ny, imgMove1.nz);
	Box32f boxMove2(0, 0, 0, imgMove2.nx, imgMove2.ny, imgMove2.nz);
	boxMove1.Offset(offset1);
	boxMove2.Offset(offset2);
	float x0, y0, z0, x1, y1, z1;
	x0 = MIN3(boxRef.x, boxMove1.x, boxMove2.x);
	y0 = MIN3(boxRef.y, boxMove1.y, boxMove2.y);
	z0 = MIN3(boxRef.z, boxMove1.z, boxMove2.z); 
	x1 = MAX3(boxRef.x + boxRef.width, boxMove1.x+ boxMove1.width, boxMove2.x+ boxMove2.width);
	y1 = MAX3(boxRef.y + boxRef.height, boxMove1.y+ boxMove1.height, boxMove2.y+ boxMove2.height);
	z1 = MAX3(boxRef.z + boxRef.depth, boxMove1.z+ boxMove1.depth, boxMove2.z+ boxMove2.depth);
	Box32i boxRes(0, 0, 0, 0, 0, z1 - z0 + 0.5);
	boxRes.width = boxRes.height = MAX(x1 - x0 + 0.5, y1 - y0 + 0.5);
	img.MallocBuffer(boxRes.width, boxRes.height, boxRes.depth);
	Point3d32i shift(-x0, -y0, -z0);
	//VolumeT<char> sum(img.nx, img.ny, img.nz);
	img.SetValue(-1024);

	#pragma omp parallel for
	for (int k=0; k<imgRef.nz; k ++){
		for (int i=0; i<imgRef.ny; i ++){
			for (int j=0; j<imgRef.nx; j ++){
				int _x = j + shift.x;
				int _y = i + shift.y;
				int _z = k + shift.z;
				if (img.IsValid(_x, _y, _z)){
					T& val = imgRef(j, i, k);
					img(_x, _y, _z) = std::max(val, img(_x, _y, _z));
					//img(_x, _y, _z) = imgRef(j, i, k);
					//sum(j, i, k) ++;
				}
			}
		}
	}
	//img.Save("c:/tst.volume");
	//return true;
	#pragma omp parallel for
	for (int k=0; k<imgMove1.nz; k ++){
		for (int i=0; i<imgMove1.ny; i ++){
			for (int j=0; j<imgMove1.nx; j ++){
				int x = j+offset1.x;
				int y = i+offset1.y;
				int z = k+offset1.z;
				int _x = x + shift.x;
				int _y = y + shift.y;
				int _z = z + shift.z;
				if (img.IsValid(_x, _y, _z)){
					T& val = img(_x, _y, _z);
					val = std::max(imgMove1(j, i, k),val);
					//sum(x, y, z) ++;
				}
			}
		}
	}
	if (&imgMove1 == &imgMove2 && offset1 == offset2){
	}else{
#pragma omp parallel for
		for (int k=0; k<imgMove2.nz; k ++){
			for (int i=0; i<imgMove2.ny; i ++){
				for (int j=0; j<imgMove2.nx; j ++){
					int x = j+offset2.x;
					int y = i+offset2.y;
					int z = k+offset2.z;
					int _x = x + shift.x;
					int _y = y + shift.y;
					int _z = z + shift.z;
					if (img.IsValid(_x, _y, _z)){
						T& val = img(_x, _y, _z);
						val = std::max(imgMove2(j, i, k),val);
						//sum(x, y, z) ++;
					}
				}
			}
		}
	}
	//for (int i=0; i<img.nx*img.ny*img.nz; i ++){
	//	if (sum.buffer[i] > 1) img.buffer[i] /= sum.buffer[i];
	//}

	return true;
#endif
}

template<typename T> inline
	bool Stitch(const VolumeT<T>& imgRef, const VolumeT<T>& imgMove, Point3d32f offset, VolumeT<T>& img)
{
	return Stitch(imgRef, Point3d32f(0, 0, 0), imgMove, offset, imgMove, offset, imgMove, offset, img);
}

template<typename T> inline
bool Stitch(const ImageT<T>& imgRef, const ImageT<T>& imgMove, Point2D32f offset, ImageT<T>& img)
{
	img.MallocBuffer(imgRef.width + imgMove.width, imgRef.height + imgMove.height);
	ImageT<int> sum(img.width, img.height);
	img.SetValue(-1024);
	for (int i=0; i<imgRef.height; i ++){
		for (int j=0; j<imgRef.width; j ++){
			if (img.IsValid(j, i)){
				img(j, i) = imgRef(j, i);
				sum(j, i) ++;
			}
		}
	}
	for (int i=0; i<imgMove.height; i ++){
		for (int j=0; j<imgMove.width; j ++){
			int x = j+offset.x;
			int y = i+offset.y;
			if (img.IsValid(x, y)){
				img(x, y) = imgMove(j, i);
				sum(x, y) ++;
			}
		}
	}
	for (int i=0; i<img.width*img.height; i ++){
		if (sum.buffer[i] > 1) img.buffer[i] /= sum.buffer[i];
	}

	return true;
}


template<typename T>
struct ProjectionImageT{
	typedef T DataType;
	typedef FrameT<T> FrameType;
	ProjectionImageT(){
		buffer = NULL;
		roi = NULL;
		step = 1;
	}
	ProjectionImageT(const ProjectionImageT& proj):buffer(NULL),roi(NULL), step(1){
		*this = proj;
	}
	ProjectionImageT& operator=(const ProjectionImageT& proj){
		if (this->MallocBuffer(proj.width, proj.height, proj.GetCount(), proj.GetStep()).IsValid()){
			memcpy(buffer, proj.Buffer(), proj.BufferSize()*sizeof(ProjectionImageT::DataType));
			if (proj.roi) this->SetRoi(*proj.roi);
		}
		return *this;
	}
//	inline bool GetRoiData(ProjectionImageT& proj, Rect32i roi) const{
//		const int x0 = MAX(0, roi.x);
//		const int y0 = MAX(0, roi.y);
//		const int x1 = MIN(roi.x + roi.width - 1, width-1);
//		const int y1 = MIN(roi.y + roi.height - 1, height-1);
//		const int w = x1 - x0 + 1;
//		const int h = y1 - y0 + 1;
//		const int count = images.size();
//		bool bRtn = proj.MallocBuffer(w, h, count, step).IsValid();
//		if (bRtn){
//#pragma omp parallel for
//			for (int i=0; i<count; i++){
//				FrameT<T> frame0, frame1;
//				bRtn = GetFrame(frame0, i);
//				VERIFY_TRUE(bRtn);
//				bRtn = proj.GetFrame(frame1, i);
//				VERIFY_TRUE(bRtn); 
//				bRtn = frame0.GetRoiData(frame1, x0, y0, x1, y1);
//				VERIFY_TRUE(bRtn); 
//			}
//		}
//		return bRtn;
//	}
	inline bool GetFrame(FrameT<T>& frame, int index) const{
		if (index >= 0 && index < images.size()){
			frame.buffer = images[index];
			frame.index = index;
			frame.width = width;
			frame.height = height;
			frame.roi = roi;
			return true;
		}
		return false;
	}
	inline FrameT<T> GetFrame(int index) const{
		FrameT<T> frame;
		if (!this->GetFrame(frame, index)){
			VERIFY_TRUE(0);
		}	
		return frame;
	}
	inline bool SetFrame(const T* buffer, int w, int h, int index){
		FrameT<T> frame;
		if (GetFrame(frame, index)){
			return frame.SetData(buffer, w, h);
		}
		return false;
	}
	inline bool SetAllFrame(const T* buffer, int w, int h){
		bool bRtn = true;
		const int count = images.size();
#pragma omp parallel for
		for (int i=0; i<count; i ++){
			VERIFY_TRUE(SetFrame(buffer, w, h, i));
		}
		return bRtn;
	}
	void Display(std::string name, const Rect32i* curRoi = NULL) const{
		for (int i=0; i<this->GetCount(); i += this->step){
			if (i != 0) ClearCurrentLine();
			FrameT<T> frame;			
			if (GetFrame(frame, i)){
				printf("image index = %06d", i);
				frame.display(name, curRoi);
				WaitKey(5);
			}
		}
		ClearCurrentLine();
		::DestroyWindow(name.c_str());
	}
	inline ProjectionImageT& Ones(){
		std::vector<T> tmp(width*height+1);
		T* pBuf = &tmp[0];
		for (int i=0; i<tmp.size(); i ++) pBuf[i] = T(1);
		VERIFY_TRUE(SetAllFrame(pBuf, width, height));
		return *this;
	}
	inline ProjectionImageT& Zero(){
		std::vector<T> tmp(width*height+1);
		T* pBuf = &tmp[0];
		VERIFY_TRUE(SetAllFrame(pBuf, width, height));
		return *this;
	}
	virtual ~ProjectionImageT(){
		if (buffer) delete []buffer;
		if (roi)    delete roi;
	}
	inline void SetRoi(Rect32i rect){
		if (roi) delete roi;
		roi = NULL;
		if (rect.width == 0 && rect.height == 0){
		}else{
			roi = new Rect32i(rect);
		}
	}
	inline ProjectionImageT& MallocBuffer(int64 w, int64 h, int64 count, int _step = 1) {
		DISPLAY_FUNCTION;
		if (w*h*count == width*height*images.size()){
		}else{
			if (buffer) delete []buffer;
			buffer = 0;
		}
		if (roi)    delete roi;
		roi = NULL;
		bool bRtn = false;
		width = w;
		height = h;
		if (width*height>0){
			if (buffer == NULL){
				int64 sz = width*height*(count + int64(1));
				//std::cout << "begin new : " << sz / 1024 / 1024 / 1024 << " G" << std::endl;
				try
				{
					buffer = new T[sz];
				}catch (...){
					std::cout<<"error : new buffer "<<sz<<std::endl;
				}
				//std::cout << "end new" << std::endl;
				memset(buffer, 0, sizeof(buffer[0])*sz);
			}
			images.resize(count);
			for (int i=0; i<count; i++) images[i] = buffer + i*width*height;
			bRtn = true;
		}
		assert(_step > 0);
		VERIFY_TRUE(bRtn);
		this->step = MAX(1, _step);
		return *this;
	}
	void FreeBuffer(){
		if (buffer) delete []buffer; buffer = 0;
		width = height = 0;
		this->images.clear();
		this->step = 0;
		roi = NULL;
	}
	bool IsValid() const{
		return (this->buffer && this->GetCount()>0)?true:false;
	}
	template<typename D> 
	inline bool LoadRawData(std::string sPath, int w, int h, int count, int _step = 1, int _startIndex = 0){
		bool bRtn = false;
		if (width == w && height == h && images.size() == count) bRtn = true;
		else bRtn = MallocBuffer(w, h, count, _step).IsValid();
		assert(this->step == _step);
		if (bRtn){
			int sz = w*h*count;
			std::vector<D> vecBuf(sz+1);
			D* buf = &vecBuf[0];
			bRtn = LoadImageFromFile<D>(sPath.c_str(), buf, width, height, count);
			if (bRtn){
				bRtn = ::Transform(buf, this->buffer, sz);
			}
		}
		return bRtn;
	}
	template<typename D>
	inline bool LoadFolder(const char* szFolder, int w, int h, int count, int _step = 1, int _startIndex = 0)
	{
		bool bRtn = false;
		if (width == w && height == h && images.size() == count) bRtn = true;
		else{
			bRtn = true;
			MallocBuffer(w, h, count, _step);
		}
		assert(this->step == _step);
		uint tm = timeGetTime();
		if (bRtn) bRtn = LoadProjectionImageFolder<D, T>(szFolder, width, height, images, this->step, _startIndex);
		tm = timeGetTime() - tm;
		printf("%s, ProjImage_Load:%s, time=%d ms\n", bRtn?"successed":"failed", szFolder, tm);
		return bRtn;
	}
	inline bool SaveToFolder(std::string strFolder) const{
		bool bRtn = false;
		MakePathDirIfNeed((strFolder + "t.t").c_str());
		const int count = images.size();
#pragma omp parallel for 
		for (int i=0; i<count; i += this->step){
			char szName[256] = "";
			sprintf(szName, "%simg%04d.raw", strFolder.c_str(), i);
			bRtn = WriteToFile<T, T>(szName, images[i], width*height);
			assert(bRtn);
		}
		printf("%s, ProjImage_Save:%s\n", bRtn?"successed":"failed",strFolder.c_str());
		return bRtn;
	}
	inline void ShiftToNoneNegative(){
		::ShiftToNoneNegative(buffer, BufferSize());
	}
	inline ProjectionImageT& log(){
		for (int i=0; i<width*height*images.size(); i ++) buffer[i] = (T)std::log(double(buffer[i]));
		return *this;
	}
	inline ProjectionImageT& exp(){
		for (int i=0; i<width*height*images.size(); i ++) buffer[i] = std::exp(buffer[i]);
		return *this;
	}
	inline ProjectionImageT& operator *=(double scale){
#pragma omp parallel for
		for (int i=0; i<width*height*images.size(); i ++) buffer[i] *= scale;
		return *this;
	}
	inline ProjectionImageT& operator /=(double scale){
		if (IsZero(scale)){
#pragma omp parallel for
			for (int i=0; i<width*height*images.size(); i ++) buffer[i] = 0;
		}else{
			*this *= double(1.0)/scale;
		}
		return *this;
	}
	inline ProjectionImageT& operator+=(double scale){
#pragma omp parallel for
		for (int i=0; i<width*height*images.size(); i ++) buffer[i] += scale;
		return *this;
	}
	inline ProjectionImageT& operator-=(double scale){
#pragma omp parallel for
		for (int i=0; i<width*height*images.size(); i ++) buffer[i] -= scale;
		return *this;
	}
	inline ProjectionImageT& operator-=(const ProjectionImageT& proj){
		const T* data = proj.Buffer();
		const int size = width*height*images.size();
#pragma omp parallel for
		for (int i=0; i<size; i ++) buffer[i] -= data[i];
		return *this;
	}
	inline ProjectionImageT& operator+=(const ProjectionImageT& proj){
		const T* data = proj.Buffer();
		const int size = width*height*images.size();
#pragma omp parallel for
		for (int i=0; i<size; i ++) buffer[i] -= data[i];
		return *this;
	}
	inline ProjectionImageT& operator/=(const ProjectionImageT& proj){
		const T* data = proj.Buffer();
		const int size = width*height*images.size();
#pragma omp parallel for
		for (int i=0; i<size; i ++){
			if (IsZero(data[i])){
				buffer[i] = 0;
			}else{
				buffer[i] /= data[i];
			}
		}
		return *this;
	}
	inline T* Buffer() const{
		return buffer;
	}
	inline int64 BufferSize() const {
		return width*height*images.size();
	}
	inline int GetCount() const{
		return (int)images.size();
	}
	inline int GetStep() const{
		return step;
	}
	inline ProjectionImageT& SetStep(int step){
		this->step = step;
		return *this;
	}
	inline ProjectionImageT& Transpose(){
		int tm = timeGetTime();
		std::vector<T> buf(width*height+1);
		T* pBuf = &buf[0];
		for (int i=0; i<this->GetCount(); i ++){
			FrameT<T> frame;
			this->GetFrame(frame, i);
			for (int y=0; y<height; y ++){
				for (int x = 0; x<width; x ++){
					pBuf[height*x+y] = frame.buffer[y*width+x];
				}
			}
			memcpy(frame.buffer, pBuf, sizeof(pBuf[0])*width*height);
		}
		std::swap(width, height);
		if (roi){
			std::swap(roi->x, roi->y);
			std::swap(roi->width, roi->height);
		}
		tm = timeGetTime() - tm;
		std::cout << "ProjectionImageT::Transpose, time=" << tm << std::endl;
		return *this;
	}
	Rect32i* roi;
	size_t width, height;
private:
	int step;
	std::vector<T*> images;
	T* buffer;
};


template<typename T>
void SegmentProjectionImage(const ProjectionImageT<T>& src, ProjectionImageT<T>& dstBigger, ProjectionImageT<T>& dstSmaller, T threshold)
{
	dstBigger.MallocBuffer(src.width, src.height, src.GetCount(), src.GetStep());
	dstSmaller.MallocBuffer(src.width, src.height, src.GetCount(), src.GetStep());
	const T* data = src.Buffer();
	T* bigger = dstBigger.Buffer();
	T* smaller = dstSmaller.Buffer();
	const int size = src.BufferSize();
#pragma omp parallel for
	for (int i=0; i<size; i++){
		if (data[i] >= threshold){
			bigger[i] = data[i];
			smaller[i] = 0;
		}else{
			bigger[i] = 0;
			smaller[i] = data[i];
		}
	}
}

typedef ProjectionImageT<float> ProjectionImage;

template <typename T>
struct op_dist{
	op_dist(T val):val(val){
	}
	T operator()(const T& x, const T& y) const{
		return val/sqrt(val*val + x*x + y*y);
	}
	const T val;
};

template <typename T>
struct op_rate{
	op_rate(T dsd, T dso):dsd(dsd),dso(dso){
	}
	T operator()(const T& x, const T& y) const{
		return (x*dsd)/(y + dso);
	}
	const T dsd, dso;
};

template<typename TA, typename TB, typename TC, typename TD> 
inline void Rotate(TA centerX, TA centerY, TB fcos, TB fsin, TC srcX, TC srcY, TD& dstX, TD& dstY)
{
	dstX = (srcX - centerX)*fcos - (srcY - centerY)*fsin + centerX;
	dstY = (srcX - centerX)*fsin + (srcY - centerY)*fcos + centerY;
}

template<typename T>
struct DetectorCoor{
	typedef T DataType;
	DetectorCoor(T du, T dv, T nu, T nv, T offset_u, T offset_v)
		:du(du), dv(dv), nu(nu), nv(nv), offset_u(offset_u), offset_v(offset_v){
			centerX = (nu-1)/2.0 + offset_u;
			centerY = (nv-1)/2.0 + offset_v;
	}
	template <typename TA, typename TB>  
	inline void mm2pixel(TA uMM, TA vMM, TB& u, TB& v) const{
		u = uMM/du;
		v = vMM/dv;
	}
	template <typename TA, typename TB>
	inline void pixel2mm(TA u, TA v, TB& uMM, TB& vMM) const{
		uMM = u*du;
		vMM = v*dv;
	}
	template <typename TA, typename TB>
	inline void xy2ij(TA x, TA y, TB& i, TB& j) const{
		i = x + centerX;
		j = centerY - y;
	}
	template <typename TA, typename TB>
	inline void ij2xy(TA i, TA j, TB& x, TB& y) const{
		x = i - centerX;
		y = centerY - j;
	}
	T du, dv;
	T nu, nv;
	T centerX, centerY;
	T offset_u, offset_v;
};

template<typename T>
struct VolumeCoor{
	typedef T DataType;
	VolumeCoor(T dx, T dy, T dz, 
		T nx, T ny, T nz):dx(dx),dy(dy),dz(dz),nx(nx),ny(ny),nz(nz){
			centerX = (nx-1.0)/2.0; centerY = (ny-1.0)/2.0; centerZ = (nz-1.0)/2.0;
	}
	template <typename TA, typename TB>
	inline void ijk2xyz(TA i, TA j, TA k, TB& x, TB& y, TB& z) const{
		x = i - centerX;
		y = centerY - j;
		z = centerZ - k;
	}
	template <typename TA, typename TB>
	inline void xyz2ijk(const TA& x, const TA& y, const TA& z, TB& i, TB& j, TB& k) const{
		i = x + centerX;
		j = centerY - y;
		k = centerZ - z;
	}
	template <typename TA, typename TB>
	inline void pixel2mm(const TA& x, const TA& y, const TA& z, TB& xmm, TB& ymm, TB& zmm) const{
		xmm = dx*x;
		ymm = dy*y;
		zmm = dz*z;
	}
	template <typename TA, typename TB>
	inline void mm2pixel(const TA& xmm, const TA& ymm, const TA& zmm, TB& x, TB& y, TB& z) const{
		x = xmm/dx;
		y = ymm/dy;
		z = zmm/dz;
	}
	T dx, dy, dz;
	T nx, ny, nz;
	T centerX, centerY, centerZ;
};

////////////////////////////////////////////////////////////////////////////////
/**
This computes an in-place complex-to-complex FFT 
x and y are the real and imaginary arrays of n=2^m points.
o(n)=n*log2(n)
dir =  1 gives forward transform
dir = -1 gives reverse transform 
Written by Paul Bourke, July 1998
FFT algorithm by Cooley and Tukey, 1965 
*/
template<typename T>
inline bool FFT(int32_t dir,int32_t m,T *x,T *y)
{
	int32_t nn,i,i1,j,k,i2,l,l1,l2;
	T c1,c2,tx,ty,t1,t2,u1,u2,z;

	/* Calculate the number of points */
	nn = 1<<m;

	/* Do the bit reversal */
	i2 = nn >> 1;
	j = 0;
	for (i=0;i<nn-1;i++) {
		if (i < j) {
			tx = x[i];
			ty = y[i];
			x[i] = x[j];
			y[i] = y[j];
			x[j] = tx;
			y[j] = ty;
		}
		k = i2;
		while (k <= j) {
			j -= k;
			k >>= 1;
		}
		j += k;
	}

	/* Compute the FFT */
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for (l=0;l<m;l++) {
		l1 = l2;
		l2 <<= 1;
		u1 = 1.0;
		u2 = 0.0;
		for (j=0;j<l1;j++) {
			for (i=j;i<nn;i+=l2) {
				i1 = i + l1;
				t1 = u1 * x[i1] - u2 * y[i1];
				t2 = u1 * y[i1] + u2 * x[i1];
				x[i1] = x[i] - t1;
				y[i1] = y[i] - t2;
				x[i] += t1;
				y[i] += t2;
			}
			z =  u1 * c1 - u2 * c2;
			u2 = u1 * c2 + u2 * c1;
			u1 = z;
		}
		c2 = sqrt((1.0 - c1) / 2.0);
		if (dir == 1)
			c2 = -c2;
		c1 = sqrt((1.0 + c1) / 2.0);
	}

	/* Scaling for forward transform */
	if (dir == -1) {
		for (i=0;i<nn;i++) {
			x[i] /= (double)nn;
			y[i] /= (double)nn;
		}
	}

	return true;
}

//supporse filter real value only, image value is 0
template<typename TypeA, typename TypeB> inline
	void filterImageRow(const std::vector<double>& filter,
	const TypeA* data_src, 
	TypeB* data_dst, 
	int width, int height)
{
	const int size = filter.size();
	assert(size > 0);

	//frame.display("filterImageColum");
	//WaitKey();

	typedef double FFT_DATA_TYPE;

	const int index = size/2 - width/2;
	const int N = log(size)/log(2) + 0.5;
#ifndef _DEBUG
#pragma omp parallel for
#endif
	for (int j=0; j<height; j ++){
		int i = 0;
		int k = 0;
		std::vector<FFT_DATA_TYPE> bufReal(size+16);
		std::vector<FFT_DATA_TYPE> bufImag(size+16);
		const double* pFilter = &filter[0];
		FFT_DATA_TYPE* pReal = &bufReal[0];
		FFT_DATA_TYPE* pImag = &bufImag[0];
		memset(pReal, 0, sizeof(*pReal)*size);
		memset(pImag, 0, sizeof(*pImag)*size);

		for (i=0; i<width; i ++) pReal[index+i] = data_src[j*width + i];
#if 1
#pragma message("filter boundary handling, constant")
		const FFT_DATA_TYPE lVal = (pReal[index+width-1] + pReal[index+width-2] + pReal[index+width-3])/3;
		const FFT_DATA_TYPE rVal = (pReal[index] + pReal[index+1] + pReal[index+2])/3;
		for (k=index+width;k<size; k++) pReal[k] = lVal;
		for (k=0;k<index; k++) pReal[k] = rVal;
#else
#pragma message("filter boundary handling, symmetric")
		//for (i=0; i < index; i ++){
		//	pReal[index - 1 - i]     = pReal[index + 1 + i];
		//	pReal[index + width + i] = pReal[index + width - 1 - i];
		//}

		//for (i=1, k=index+width;k<size; k++, i++) pReal[k] = pReal[index+width-i];
		//for (i=1, k=index-1;k>=0; k--, i++)       pReal[k] = pReal[index      +i];
#endif
		bool brtn = FFT(1, N, pReal, pImag);
		assert(brtn);

		//filter
		for (i=0; i<size; i ++){
			pReal[i] *= FFT_DATA_TYPE(pFilter[i]);
			pImag[i] *= FFT_DATA_TYPE(pFilter[i]);
			//const double& filterReal = filter[i];
			//const double filterImag = 0; //filter[i];
			//double fr = pReal[i]*filterReal - pImag[i]*filterImag;
			//double fi = pReal[i]*filterImag + pImag[i]*filterReal;
			//pReal[i] = fr;
			//pImag[i] = fi;
		}

		brtn =FFT(-1, N, pReal, pImag);
		assert(brtn);

		for (i=0; i<width; i ++){
			data_dst[j*width + i] = pReal[index+i];
		}
		//image.display("image-filter");
	}
}

template<typename TypeA, typename TypeB>
void filterImageRow(const std::vector<double>& filter, const FrameT<TypeA>& frame, ImageT<TypeB>& image)
{
	filterImageRow(filter,frame.data,image.buffer, frame.width, frame.height);
}

template<typename TypeA, typename TypeB>
void filterImageRow(const std::vector<double>& filter, const ImageT<TypeA>& frame, ImageT<TypeB>& image)
{
	filterImageRow(filter,frame.buffer, image.buffer, frame.width, frame.height);
}


template<typename TypeA, typename TypeB>
void filterImageRow(const FrameT<TypeA>& frame, const std::vector<double>& filter, ImageT<TypeB>& image)
{
	assert(frame.width == image.width && frame.height == image.height);
	const int size = filter.size();
	assert(size > 0);
	int width = frame.width;
	int height = frame.height;

	std::vector<double> bufReal(size+1);
	std::vector<double> bufImag(size+1);
	double* pReal = &bufReal[0];
	double* pImag = &bufImag[0];

	//frame.display("filterImageColum");
	//WaitKey();

	const int index = size/2 - width/2;
	const int N = log(size)/log(2) + 0.5;
	int i, j;
	for (j=0; j<height; j ++){
		memset(pReal, 0, sizeof(*pReal)*size);
		memset(pImag, 0, sizeof(*pImag)*size);

		for (i=0; i<width; i ++){
			pReal[index+i] = frame.data[j*width + i];
		}
		bool brtn = FFT(1, N, pReal, pImag);
		assert(brtn);

		//filter
		for (i=0; i<size; i ++){
			pReal[i] *= filter[i];
			pImag[i] *= filter[i];
		}

		brtn =FFT(-1, N, pReal, pImag);
		assert(brtn);

		for (i=0; i<width; i ++){
			image.buffer[j*width + i] = pReal[index+i];
		}
		//image.display("image-filter");
	}
}

template<typename TypeA, typename TypeB>
void filterImageColum(const std::vector<double>& filter,
					  const TypeA* data_src, 
					  TypeB* data_dst, 
					  int width, int height)
{
	const int size = filter.size();
	assert(size > 0);

	std::vector<double> bufReal(size+1);
	std::vector<double> bufImag(size+1);
	double* pReal = &bufReal[0];
	double* pImag = &bufImag[0];

	const int index = size/2 - height/2;
	const int N = log(size)/log(2) + 0.5;
	//#pragma omp parallel for num_threads(16)
	for (int i=0; i<width; i ++){
		int j;
		memset(pReal, 0, sizeof(*pReal)*size);
		memset(pImag, 0, sizeof(*pImag)*size);

		for (j=0; j<height; j ++){
			pReal[index+j] = data_src[j*width + i];
		}
		bool brtn = FFT(1, N, pReal, pImag);
		assert(brtn);

		//filter
		for (j=0; j<size; j ++){
			pReal[j] *= filter[j];
			pImag[j] *= filter[j];
		}

		brtn =FFT(-1, N, pReal, pImag);
		assert(brtn);

		for (j=0; j<height; j ++){
			data_dst[j*width + i] = pReal[index+j];
		}
	}
}

template<typename TypeA, typename TypeB>
void filterImageColum(const std::vector<double>& filter, const FrameT<TypeA>& frame, ImageT<TypeB>& image)
{
	assert(frame.width == image.width && frame.height == image.height);
	filterImageColum(filter, frame.data, image.buffer, frame.width, frame.height);
}

template<typename TypeA, typename TypeB>
void filterImageColum(const std::vector<double>& filter, const ImageT<TypeA>& src, ImageT<TypeB>& dst)
{
	assert(src.width == dst.width && src.height == dst.height);
	filterImageColum(filter, src.buffer, dst.buffer, src.width, src.height);
}

template<typename TypeA, typename TypeB, typename TypeC>
void convolutionImageRow(const std::vector<TypeC>& filter,
						 const TypeA* data_src, 
						 TypeB* data_dst, 
						 int width, int height)
{
	memcpy(data_dst, data_src, sizeof(*data_dst)*width*height);
	const int radius = filter.size()/2;
#pragma omp parallel for
	for (int i=0; i<height; i++){
		for (int j=radius; j<width-radius; j ++){
			const TypeA* psrc = data_src + i*width + j;
			TypeB* pdst = data_dst + i*width + j;
			double sum = 0;
			int idx = 0;
			for (int k=-radius; k<=radius; k++, idx++){
				sum += psrc[k]*filter[idx];
			}
			*pdst = sum/idx;
		}
	}
}

template<typename TypeA, typename TypeB, typename TypeC>
void convolutionImageColume(const std::vector<TypeC>& filter,
							const TypeA* data_src, 
							TypeB* data_dst, 
							int width, int height)
{
	memcpy(data_dst, data_src, sizeof(*data_dst)*width*height);
	const int radius = filter.size()/2;

#pragma omp parallel for
	for (int i=radius; i<height-radius; i++){
		for (int j=0; j<width; j ++){
			const TypeA* psrc = data_src + i*width + j;
			TypeB* pdst = data_dst + i*width + j;
			double sum = 0;
			int idx = 0;
			for (int k=-radius; k<=radius; k++, idx++){
				sum += psrc[k*width]*filter[idx];
			}
			*pdst = sum/idx;
		}
	}
}

template<typename TypeA, typename TypeB, typename TypeC>
void convoluteImage(const std::vector<TypeC>& filterRow,
					const std::vector<TypeC>& filterCol,
					const TypeA* data_src, 
					TypeB* data_dst, 
					int width, int height)
{
	std::vector<TypeA> vecDataFilterRow(width*height+1);
	TypeA* pRow = &vecDataFilterRow[0];
	convolutionImageRow(filterRow, data_src, pRow, width, height);
	convolutionImageColume(filterCol, pRow, data_dst, width, height);
}


inline int nextpow2(int n)
{
	unsigned m = n;
	int i=0;
	for (i=0; m>0;i++) m>>=1;
	m = 1<<(i);
	return m;
}

inline std::vector<double> filter(std::string strType, int len, int d)
{
	int i, j;
	std::vector<double> vec, filt, w;
	int order =nextpow2(len);
	//if (order/2 < len) order <<= 1;

	filt.resize(order/2+1);
	for (i=0; i<filt.size(); i++) filt[i] = i*1.0*2/order;
	w.resize(filt.size());
	for (i=0; i<w.size(); i++) w[i] = 2*PI*i/order;
	if (strType == "none"){
	}else if (strType == "ram-lak"){
	}else if (strType == "shepp-logan"){
		//filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)));
		for (i=1; i<filt.size(); i++){
			filt[i] *= sin(w[i])/(2*d)/(w[i]/(2*d));
		}
	}else if (strType == "cosine"){
		//filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)));
		for (i=1; i<filt.size(); i++){
			filt[i] *= cos(w[i]/(2.0*d));
		}
	}else if (strType == "hamming"){
		// The Hanning window is defined as:
		// w(k) = alpha - (1-alpha)*cos(2 pi k/ N), k = 0,1,...,N-1
		// where alpha=0.5 for the Hanning, and alpha=0.54 for the Hamming window.
		//
		//filt(2:end) = filt(2:end) .* (.54 + .46 * cos(w(2:end)/d));
		for (i=1; i<filt.size(); i++){
			filt[i] *= (0.54 + 0.46*cos(w[i])/d);
		}
	}else if (strType == "hann"){
		//filt(2:end) = filt(2:end) .*(1+cos(w(2:end)./d)) / 2;
		for (i=1; i<filt.size(); i++){
			filt[i] *= (1.0 + cos(w[i]/d))/2.0;
		}
	}else{
		assert(0);
	}

	vec.resize(filt.size()*2-2);
	std::copy(filt.begin(), filt.end(), vec.begin());
	for (j=0, i=filt.size(); i<vec.size(); i++, j++) vec[i] = filt[filt.size()-2-j];
	for (i=0; i<vec.size(); i ++) {
		if (vec[i] > PI*d) vec[i] = 0;
	}
	return vec;
}

template<typename T>
inline void filter(std::string strType, int len, int d, std::vector<T>& vec){
	int i, j;
	std::vector<double> filt, w;
	int order =nextpow2(len);
	//if (order/2 < len) order <<= 1;

	filt.resize(order/2+1);
	for (i=0; i<filt.size(); i++) filt[i] = i*1.0*2/order;
	w.resize(filt.size());
	for (i=0; i<w.size(); i++) w[i] = 2*PI*i/order;
	if (strType == "none"){
	}else if (strType == "ram-lak"){
	}else if (strType == "shepp-logan"){
		//filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)));
		for (i=1; i<filt.size(); i++){
			filt[i] *= sin(w[i])/(2*d)/(w[i]/(2*d));
		}
	}else if (strType == "cosine"){
		//filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)));
		for (i=1; i<filt.size(); i++){
			filt[i] *= cos(w[i]/(2.0*d));
		}
	}else if (strType == "hamming"){
		// The Hanning window is defined as:
		// w(k) = alpha - (1-alpha)*cos(2 pi k/ N), k = 0,1,...,N-1
		// where alpha=0.5 for the Hanning, and alpha=0.54 for the Hamming window.
		//
		//filt(2:end) = filt(2:end) .* (.54 + .46 * cos(w(2:end)/d));
		for (i=1; i<filt.size(); i++){
			filt[i] *= (0.54 + 0.46*cos(w[i])/d);
		}
	}else if (strType == "hann"){
		//filt(2:end) = filt(2:end) .*(1+cos(w(2:end)./d)) / 2;
		for (i=1; i<filt.size(); i++){
			filt[i] *= (1.0 + cos(w[i]/d))/2.0;
		}
	}else{
		assert(0);
	}

	vec.resize(filt.size()*2-2);
	std::copy(filt.begin(), filt.end(), vec.begin());
	for (j=0, i=filt.size(); i<vec.size(); i++, j++) vec[i] = filt[filt.size()-2-j];
}

template<typename TypeA, typename TypeB>
void MedianFilter(const TypeA* src, TypeB*dst, int width, int height, int diameter=3)
{
	const int radius = diameter/2;
	const int count = diameter*diameter;
#pragma omp parallel for
	for (int i=0; i<height; i ++){
		for (int j=0; j<width; j++){
			std::vector<TypeA> tmp(count+1);
			TypeA* pCur = &tmp[0];
			int idx = 0;
			for (int y=-radius; y<=radius; y++){
				for (int x=-radius; x<=radius; x++,idx++){
					int _x = j + x;
					int _y = i + y;
					if (_x >=0 && _x < width && _y>=0 && _y < height){
						pCur[idx] = (src + width*_y)[_x];
						//std::cout<<pCur[idx]<<" ";
					}
				}
			}
			//std::cout<<std::endl;
			if (idx > 0){
				std::sort(pCur, pCur+idx, std::less<TypeA>());
				dst[i*width + j] = pCur[idx/2];
			}
		}
	}
}

struct GeometryInformation{
	GeometryInformation(){
		init();
	}
	void init(){
		nx = ny = nz = 512;
		//nx = ny = 384;
		//nz = 256;
		//dx = 0.157000*2; 
		//dy = 0.157000*2; 
		//dz = 0.146000*2;
		//dx =  dy =  dz = 0.157000;
		dx =  dy =  dz = 0.157000;
		//dx *= 4;
		//dy *= 4;
		nu = 608; //rows
		nv = 616; //colums
		du = 0.2;  dv = 0.2;

		DSO = 468;   //source to object
		DSD = 700.0; //source to detetor  //mm

		offset_nu = offset_nv = 0;

		sx = dx*nx; sy = dy*ny; sz = dz*nz;
		su = du*nu; 
		sv = dv*nv;
	}
	int nx, ny, nz;
	double dx, dy, dz;
	double sx, sy, sz;

	int nu, nv;
	double du, dv;
	double su, sv;

	double DSD, DSO;

	double offset_nu, offset_nv;   //pixel unit
};

struct ScannerInformation{
	ScannerInformation(){
		init();
	}
	void init(){
		direction = -1;
		scan_angle = 360; //360;
		nProj = 512;       //512;
		step = 1;
		vec_deg.resize(nProj);
		for (int i=0; i<nProj; i ++) vec_deg[i] = i *scan_angle/double(nProj);
		volume_postion_angle = -PI*45.0/180.0;
	}
	int    direction;
	double scan_angle;
	int    nProj;
	std::vector<double> vec_deg;
	double volume_postion_angle;
	int step;
};

struct Parameter;
inline Matrix2dReal4x4 CalculateProjMat(const Parameter& para, double angle_rad);
inline Matrix2dReal4x4 CalculateProjMatKJI(const Parameter& para, double angle_rad);

static const char s_FILTER_TYPE[6][73] = {
	"none",
	"ram-lak",
	"shepp-logan",
	"cosine",
	"hamming",
	"hann",
};
struct Parameter {
	Parameter() {
		{
			nx = ny = nz = 512;
			dx =  dy =  dz = 0.157000;
			dz = 0.146;
			nu = 608;  //rows
			nv = 616;  //colums
			du = 0.2; dv = 0.2;
#pragma message("scale the parameter by chenpeng")

			SOD = 468;    //source to object
			SID = 700.0;  //source to detetor  //mm

			direction = -1;
			scan_angle = 360;  //360;
			nProj = 512;      //512;

			offset_nu = 0;  //302.90763470530510 - (nu-1)/2;
			offset_nv = 0;

			offset_su = offset_nu*du;
			offset_sv = offset_nv*dv;

			step = 1;

			//volume_postion_angle = -PI*45.0/180.0;
			volume_postion_angle = -PI * 48.8671875 / 180.0;
			//volume_postion_angle = 0;

			compute_device = 1;

			str_filter = "shepp-logan";
			str_main_folder = "";
		}
		init();
	}
#define STR_CMP(_str) (strncmp(p0, #_str, strlen(#_str)) == 0 && strlen(#_str)) && (strlen(p1) > 0)
#define STR_VAL(_str)

	bool LoadParameter(const char* szPath = NULL) {
		Parameter& para = *this;
		std::fstream fs;
		fs.open(szPath, std::fstream::in);
		if (!fs.is_open()) return false;
		while (!fs.eof()) {
			char szTxt[1024 * 4] = "";
			fs.getline(szTxt, sizeof(szTxt));

			char* p0 = szTxt;
			char* p1 = szTxt + strlen(szTxt) - 1;
			while (*p1 == ' ') {
				*p1 = 0; p1--;
			}
			while (*p0 == ' ') p0++;
			if (*p0 == '#') continue;
			p1 = strchr(p0, '='); if (!p1) continue; p1++; while (*p1 == ' ') p1++;
			if (0) {
			}
			else if (STR_CMP(volume_postion_angle)) { para.volume_postion_angle = atof(p1); }
			else if (STR_CMP(nx))          { para.nx = atoi(p1); }
			else if (STR_CMP(ny))          { para.ny = atoi(p1); }
			else if (STR_CMP(nz))          { para.nz = atoi(p1); }
			else if (STR_CMP(dx))          { para.dx = atof(p1); }
			else if (STR_CMP(dy))          { para.dy = atof(p1); }
			else if (STR_CMP(dz))          { para.dz = atof(p1); }
			else if (STR_CMP(nu))          { para.nu = atoi(p1); }
			else if (STR_CMP(nv))          { para.nv = atoi(p1); }
			else if (STR_CMP(du))          { para.du = atof(p1); }
			else if (STR_CMP(dv))          { para.dv = atof(p1); }
			else if (STR_CMP(offset_nu)) { para.offset_nu = atof(p1); }
			else if (STR_CMP(offset_nv)) { para.offset_nv = atof(p1); }
			else if (STR_CMP(SID))       { para.SID = atof(p1); }
			else if (STR_CMP(SOD)) { para.SOD = atof(p1); }
			else if (STR_CMP(rot_dir)) { para.direction = atoi(p1); }
			else if (STR_CMP(angle)) { para.scan_angle = atof(p1); }
			else if (STR_CMP(proj)) { para.nProj = atoi(p1); }
			else if (STR_CMP(filter)) { para.str_filter = s_FILTER_TYPE[atoi(p1)]; }
			else if (STR_CMP(compute_device)) { para.compute_device = 1 << atoi(p1); }
			else if (STR_CMP(input_data)) { para.str_proj_img_folder = p1; }
			//else if (STR_CMP(data_type)) { sysPara.data_type = atof(p1); }
			//else if (STR_CMP(file_type)) { sysPara.file_type = atof(p1); }
			else if(STR_CMP(output_data)) {para.str_working_folder = p1; }
			//else if (STR_CMP(device_name)) { strcpy(para.sysInfo.szDeviceName, p1); }
			else if(STR_CMP(batch_count)) { para.nBatchCount = atoi(p1); }
			else if(STR_CMP(verbose))     { para.verbose = atoi(p1); }
			//else if (STR_CMP(optimize_type)) { sysPara.optimize_type = atoi(p1); }
			//else if (STR_CMP(fa)) { para.sysInfo.fa = atof(p1); }
			//else if (STR_CMP(fb)) { para.sysInfo.fb = atof(p1); }
		}
		return true;
	}
	void init(const char* szPath = NULL) {
		str_proj_img_folder = "D:/CT_Scanner/proj-img/";
		str_working_folder = "D:/CT_Scanner/3dCT_10/";		
		if (IsPathExisted(szPath)) {
			VERIFY_TRUE(LoadParameter(szPath));
		}

		//str_filter = "none";
		//str_filter = "shepp-logan";
		//str_filter = "hamming";
		//str_filter = "cosine";
		//str_filter = "hann";

		vec_deg.resize(nProj);
		for (int i = 0; i < nProj; i++) vec_deg[i] = i *scan_angle / double(nProj);
		//2/param.du*(2*pi/param.nProj)/2*(param.DSD/param.DSO)
		const int STEP=1;
		AngularAndrampFactor = 1.0 / 2.0 / this->du*(2.0*PI / (this->nProj / STEP)) / 2.0*(this->SID / this->SOD);
		//AngularAndrampFactor = DegToRad(360/(this->nProj/this->step)) * this->DSD / (2. * this->DSO)/this->du;
		//AngularAndrampFactor = 1/double(this->nProj/this->step);

		sx = dx*nx; sy = dy*ny; sz = dz*nz;
		su = du*nu; 
		sv = dv*nv;

		CalculateCosTable();
		this->fft_kernel_real = filter(str_filter, nu, 1);
		fft_kernel_real_f32.resize(fft_kernel_real.size());
		for (int i = 0; i < fft_kernel_real.size(); i++) fft_kernel_real_f32[i] = fft_kernel_real[i];

		vec_rot_mat_kji.resize(16*nProj);
		for (int i = 0; i < nProj; i++) {
			const Matrix2dReal4x4 mat_proj = CalculateProjMatKJI(*this, direction*vec_deg[i] / 180.0*PI + volume_postion_angle);	
			memcpy(&vec_rot_mat_kji[16*i], mat_proj.data, sizeof(mat_proj.data));
		}
	}

	bool CalculateCosTable()
	{
		Parameter& para = *this;
		para.cos_image_table.cos_sigma.MallocBuffer(para.nu, para.nv);
		para.cos_image_table.csc_sigma.MallocBuffer(para.nu, para.nv);

		const VolumeCoor<float> volCor(para.dx, para.dy, para.dz, para.nx, para.ny, para.nz);
		const DetectorCoor<float> detectorCor(para.du, para.dv, para.nu, para.nv, para.offset_nu, para.offset_nv);

		//const double rampFactor = para.DSD / (2. * para.DSO);

		const int size = para.nu*para.nv;
#pragma omp parallel for
		for (int sz = 0; sz < size; sz++) {
			const int i = sz / para.nu;   //y
			const int j = sz % para.nu;   //x
			float uMM, vMM;
			{
				float u, v;
				detectorCor.ij2xy(j, i, u, v);
				detectorCor.pixel2mm(u, v, uMM, vMM);
			}
			float dis = sqrt(uMM*uMM + vMM*vMM + para.SID*para.SID);
			float w = para.SID / dis;
#if 1
			w *= para.SOD*para.SOD*para.AngularAndrampFactor;
#endif
			para.cos_image_table.cos_sigma(j, i) = w;
			para.cos_image_table.csc_sigma(j, i) = 1. / w;
		}
		{
			//Forward Projection Detector xyz table in MM
			vec_rot_det_z_tbl.resize(this->nv);
			vec_rot_det_xy_tbl.resize(this->nProj);
#pragma omp parallel for
			for (int s = 0; s < this->nProj; s++) {
				double angle_rad = -(para.direction*para.vec_deg[s] / 180.0*PI + para.volume_postion_angle);
				const double fcos = cos(angle_rad);
				const double fsin = sin(angle_rad);
				const double det_yMM = this->SID - this->SOD;			
				const DetectorCoor<double> detectorCor(para.du, para.dv, para.nu, para.nv, para.offset_nu, para.offset_nv);
				vec_rot_det_xy_tbl[s].resize(this->nu);
				double u, v, uMM, vMM;
				for (int i = 0; i < this->nu; i++) {
					const int j = 0;
					Point2D64f rot_pt1_mm;
					//detector

					detectorCor.ij2xy(i, j, u, v);
					detectorCor.pixel2mm(u, v, uMM, vMM);
					Vector3<double> pt1_mm(uMM, det_yMM, vMM);
					::Rotate(double(0), double(0), fcos, fsin, pt1_mm.x, pt1_mm.y, rot_pt1_mm.x, rot_pt1_mm.y); 
					vec_rot_det_xy_tbl[s][i] = Point2D32f(rot_pt1_mm.x, rot_pt1_mm.y);
				}
				if (s == 0) {
					for (int j = 0; j < this->nv; j++) {
						const int i = 0;
						detectorCor.ij2xy(i, j, u, v);
						detectorCor.pixel2mm(u, v, uMM, vMM);
						vec_rot_det_z_tbl[j] = vMM;
					}
				}			
			}
		}
		{
			vec_rot_mat_ijk.resize(4*4*para.nProj);
			vec_rot_mat_ijk3x4.resize(3 * 4*para.nProj);
			for (int i = 0; i < nProj; i++){
				double angle_rad = para.direction*para.vec_deg[i] / 180.0*PI + para.volume_postion_angle;
				Matrix2dReal4x4 mat = CalculateProjMat(para, angle_rad);
				float* p0 = &vec_rot_mat_ijk[0] + 4 * 4*i;
				float* p1 = &vec_rot_mat_ijk3x4[0] + 3 * 4*i;
				memcpy(p0, mat.data, sizeof(p1[0]) * 4 * 4);
				memcpy(p1, mat.data, sizeof(p1[0]) * 3 * 4);
			}
		}
		return true;
	}
	void Display() const{
		//cos_image_table.cos_sigma.display("cos_image_table-cos_sigma");
		//cos_image_table.csc_sigma.display("cos_image_table-csc_sigma");
		printf("proj_img_folder=%s\n", str_proj_img_folder.c_str());
		printf("working_folder=%s\n", str_working_folder.c_str());
		printf("mask_folder=%s\n", str_mask_folder.c_str());
		printf("image: [nu(%d), nv(%d)], [du(%.3f), dv(%.3f)]\n", nu, nv, du, dv);
		printf("volume: [nx(%d), ny(%d), nz(%d)] [dx(%f), dy(%f), dz(%f)] [offset nu(%f) nv(%f)]\n", nx, ny, nz, dx, dy, dz, offset_nu, offset_nv);
		printf("cos.img [nu(%d), nv(%d)], flt.size=%d\n", cos_image_table.cos_sigma.width, cos_image_table.cos_sigma.height, fft_kernel_real_f32.size());
		printf("projection: count=(%d), scan_angle(%.3f), rot_direction(%d)\n", nProj, scan_angle, direction);
		printf("volume_postion_angle(%.3f), step(%d), filter(%s)\n", volume_postion_angle / PI * 180.0, step, str_filter.c_str());
		printf("-------------------------------------------------\n");
	}
	std::string GetProjPath(int index) const{
		VERIFY_TRUE(index >= 0 && index < nProj);
		int idx = index * step;
		return StringFormat("%simg%04d.raw", str_proj_img_folder.c_str(), idx);
	}
	double volume_postion_angle;
	int step;
	int64 nx, ny, nz;     //pixel unit
	double sx, sy, sz;    //mm

	//detector setting, according to varian trilogy obi(real size)
	double su, sv;        //mm

	//detector rotation shift (real size)
	double offset_su, offset_sv;

	double offset_nu, offset_nv;

	//the real detector panel pixel density(number of pixels)
	int64 nu, nv;        //pixel unit

	double dx, dy, dz, du, dv;

	//X-ray source & detector setting
	double SID;           //distance source to detector
	double SOD;           //X-ray source to object axis distance 

	//angle setting
	int64 direction;    
	double scan_angle;    //
	std::vector<double> vec_deg;
	int64 nProj;
	int compute_device;
	struct {
		ImageT<float> cos_sigma, csc_sigma;
	}cos_image_table;
	double AngularAndrampFactor;
	std::string str_filter;
	std::vector<double> fft_kernel_real;
	std::vector<float>  fft_kernel_real_f32;
	std::string str_main_folder, str_proj_img_folder, str_working_folder, str_mask_folder;
	std::vector<std::vector<Point2D32f>> vec_rot_det_xy_tbl;
	std::vector<float>                   vec_rot_det_z_tbl;
	std::vector<float>                   vec_rot_mat_ijk3x4, vec_rot_mat_ijk, vec_rot_mat_kji;  
	int64 verbose;
	int64 nBatchCount;
	//For fast CPU calculation
	//std::vector<double> vecXs, vecYs, vecZs, vecUs, vecVs;
};


inline Matrix2dReal4x4 CalculateProjMat(const Parameter& para, double angle_rad)
{
	typedef Matrix2dReal4x4::DataType T;
	const T fcos = cos(angle_rad);
	const T fsin = sin(angle_rad);

	Matrix2dReal4x4 mat0(
		1,  0,  0, -(para.nx-1)/2.f,
		0, -1,  0,  (para.ny-1)/2.f,
		0,  0, -1,  (para.nz-1)/2.f,
		0,  0,  0,                1
		);
	Matrix2dReal4x4 mat1(
		para.dx, 0,       0,       0,
		0,       para.dy, 0,       0,
		0,       0,       para.dz, 0,
		0,       0,       0,       1.
		);
	Matrix2dReal4x4 mat2(
		fcos,  -fsin,       0,       0,
		fsin,   fcos,       0,       0,
		0,         0,       1.,       0,
		0,         0,       0,       1.
		);
	Matrix2dReal4x4 mat3(
		1,       0,       0,       0,
		0,       0,      -1,       0,
		0,       1,       0,       para.SOD,
		0,       0,       0,       1.
		);
	Matrix2dReal4x4 mat4(
		para.SID,  0,             (para.nu-1)/2.f*para.du,       0,
		0,         para.SID,      (para.nv-1)/2.f*para.dv,       0,
		0,         0,              1,                             0,
		0,         0,              0,                             1
		);
	Matrix2dReal4x4 mat5(
		1.f/para.du,  0,             0,       0,
		0,         1.f/para.dv,      0,       0,
		0,         0,                1,       0,
		0,         0,                0,       1
		);
	Matrix2dReal4x4 mat = mat5*mat4*mat3*mat2*mat1*mat0;

	return mat;

}

inline Matrix2dReal4x4 CalculateProjMatKJI(const Parameter& para, double angle_rad)
{
	Matrix2dReal4x4 mat_0(
		0,       0,       1,              0,
		0,      -1,       0,      para.ny-1,
		-1,      0,       0,      para.nz-1,
		0,       0,       0,              1.
		);
	Matrix2dReal4x4 mat = CalculateProjMat(para, angle_rad)*mat_0;

	return mat;

}


template<typename _Tp0, typename _Tp1> inline
float NCC(const _Tp0* vec0, const _Tp1* vec1, int size)
{
	assert(size > 0);	
	float ncc = 0;
	if (size == 0) return ncc;

	float mean0 = 0;
	float mean1 = 0;
	for (int i=0; i<size; i ++){
		mean0 += vec0[i];
		mean1 += vec1[i];
	}
	mean0 /= size;
	mean1 /= size;

	float x = 0, y0 = 0, y1 = 0;
	for (int i=0; i<size; i ++){
		float s0 = vec0[i] - mean0;
		float s1 = vec1[i] - mean1;
		x += s0*s1;
		y0 += s0*s0;
		y1 += s1*s1;
	}
	ncc = x/(sqrt(y0) * sqrt(y1));
	return ncc;
}



template<typename _TpRef, typename _TpRec>
inline void CheckImageQuality(const VolumeT<_TpRef>& ref, const VolumeT<_TpRec>& recon,
							  double refLowerThreshold, double refUpperThreshold,
							  double snrThreshold = 26.0, double errorPerPixelThreshold = 0.03)
{
	typedef double ErrorType;
  ErrorType TestError = 0.;
  ErrorType EnerError = 0.;

  VERIFY_TRUE(ref.nx == recon.nx && ref.ny == recon.ny && ref.nz == recon.nz);

  const int size = ref.nx*ref.ny*ref.nz;
  for (int i=0; i<size ; i ++){
	  TestError += std::abs(recon.buffer[i] - ref.buffer[i]);
	  EnerError += std::pow(recon.buffer[i] - ref.buffer[i], 2);
  }

  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/size;
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError/size;
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20*log10(2.0) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (2.0-ErrorPerPixel)/2.0;
  std::cout << "QI = " << QI << std::endl;

  // Checking results
  if (ErrorPerPixel > errorPerPixelThreshold){
    std::cerr << "Test Failed, Error per pixel not valid! "<< ErrorPerPixel << " instead of 0.03." << std::endl;
   // exit( EXIT_FAILURE);
  }
  if (PSNR < snrThreshold){
    std::cerr << "Test Failed, PSNR not valid! "<< PSNR << " instead of 26" << std::endl;
   // exit( EXIT_FAILURE);
  }
}

template<typename _Tp0, typename _Tp1> inline
bool CheckConsistency(const std::vector<_Tp0>& vec0, std::vector<_Tp1>& vec1){
		if (vec0.size() != vec1.size()) return false;
		for (int i=0; i<vec0.size(); i ++){
			if (vec0[i] != vec1[i]) return false;
		}
		return true;
}

template<typename _TpPoint>
inline bool GetCalXY(int width, int height, std::vector<_TpPoint>& vecPos, std::vector<_TpPoint>& vecMsk){
	vecPos.clear();
	vecMsk.clear();
	float cx = (width-1)/2.0;
	float cy = (height-1)/2.0;
	//std::vector<cl_int2> vec;
	for (int i=0; i<height; i ++){
		for (int j=0; j<width; j ++){
			float dx = i - cx;
			float dy = j - cy;
			float dis = sqrt(dx*dx + dy*dy);
			_TpPoint pos;
			pos.x = j;
			pos.y = i;
			if (dis <= width / 2.0)
				vecPos.push_back(pos);
			else
				vecMsk.push_back(pos);
		}
	}
	return vecPos.empty()==false && vecMsk.empty() == false;
}

template<typename _TpPoint>
inline std::vector<_TpPoint> GetValideXY(int width, int height){
	std::vector<_TpPoint> vec_pos, vec_msk;
	VERIFY_TRUE(GetCalXY(width, height, vec_pos, vec_msk));
	return vec_pos;
}
#if 0 //working
#endif

template<typename T> 
	struct SafeList {
		SafeList() : interval(std::chrono::microseconds(1)) {
		}
		virtual ~SafeList() {
			m_arr.clear();
		}
		inline void push_back(const T& val) {
			while (!m_mtx.try_lock()) std::this_thread::sleep_for(interval);
			//std::lock_guard<std::mutex> lock(m_mtx);
			m_arr.push_back(val);
			m_mtx.unlock();
		}
		inline void push_back(const std::vector<T>& vec) {
			while (!m_mtx.try_lock()) std::this_thread::sleep_for(interval);
			//std::lock_guard<std::mutex> lock(m_mtx);
			for (auto it=vec.begin(); it != vec.end(); it ++)
				m_arr.push_back(*it);
			m_mtx.unlock();
		}		
		inline void pop_front() {
			while (!m_mtx.try_lock()) std::this_thread::sleep_for(interval);
			//std::lock_guard<std::mutex> lock(m_mtx);
			m_arr.pop_front();
			m_mtx.unlock();
		}
		inline bool get_front(T* pVal) {
			//DISPLAY_FUNCTION;
			bool bRtn = false;
			//std::lock_guard<std::mutex> lock(m_mtx);
			//m_mtx.lock();
			while (!m_mtx.try_lock()) std::this_thread::sleep_for(interval); 
			if (m_arr.size() > 0) {
				if (pVal) {
					const int sz = m_arr.size();
					*pVal = m_arr.front();
					VERIFY_TRUE(m_arr.size() > 0);
					m_arr.pop_front();
					VERIFY_TRUE(m_arr.size()+1 == sz); 
					bRtn = true;
				}
			}
			m_mtx.unlock();
			return bRtn;
		}
		inline virtual void clear() {
			std::lock_guard<std::mutex> lock(m_mtx);
			m_arr.clear();
		}
		inline int size() {
			int sz = 0;
			//std::lock_guard<std::mutex> lock(m_mtx);
			sz = m_arr.size();
			return sz;
		}
	protected:
		const std::chrono::microseconds interval;
		std::mutex m_mtx;
		std::list<T> m_arr;
	};



template<typename T>
	static float ToGB(T x) {
		return double(x) / 1024 / 1024 / 1024;
	}

inline unsigned long long getTotalSystemMemory()
{
#ifdef _WIN32
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	return status.ullTotalPhys;
#else
	unsigned long long pages = sysconf(_SC_PHYS_PAGES);
	unsigned long long page_size = sysconf(_SC_PAGE_SIZE);
	return pages * page_size;
#endif
}

#endif //__Type_H
