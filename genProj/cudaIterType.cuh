#ifndef __cudaIterType_H
#define __cudaIterType_H

#include "../common/cudaLib.cuh"
#include "cuda_helper_math.cuh"
#include "../common/type.h"

#define DevArray3D GpuArray3D
#define DevData GpuData

template <typename T> 
__device__ __host__ __forceinline__ T cuNormize(T x){
	return x/sqrt(dot(x, x));
}

__device__ __host__ __forceinline__
	int UpDivide(int x, int y){
		return (x+y-1)/y;
}

template<typename _Tp0, typename _Tp1> __device__ __forceinline__  
	bool IsInBox(_Tp0 i, _Tp0 j, _Tp1 _w, _Tp1 _h){
		return (i>=0 && i<=(_w-1) && j>=0 && j<=(_h-1))?true:false;
}

template<typename _Tp0, typename _Tp1> __device__ __forceinline__  
	bool IsInBox(_Tp0 i, _Tp0 j, _Tp0 k, _Tp1 _w, _Tp1 _h, _Tp1 _d){
		return (i>=0 && i<=(_w-1) && j>=0 && j<=(_h-1) && k>=0 && k<=(_d-1))?true:false;
}

template<typename T> __forceinline__ __host__ __device__
	T& cuPointValue(T* data, int nx, int ny, int nz, int x, int y, int z){
		assert(nx>0 && ny > 0 && nz > 0 && x>=0 && y>= 0 && z >= 0 && x<nx && y<ny && z < nz);
		return (data + nx*ny*z)[y*nx+x];
}

template<typename T> __forceinline__ __host__ __device__
	T& cuPointValue(T* data, int nx, int ny, int x, int y){
		assert(nx > 0 && x >= 0 && x <= nx -1);
		assert(ny > 0 && y >= 0 && y <= ny -1);
		return (data + nx*y)[x];
}

template<typename T> __forceinline__ __host__ __device__
	T& cuPointValue(T* data, int nx, int x){
		assert(nx > 0 && x >= 0 && x <= nx -1);
		return data[x];
}




template<typename TA, typename TB, typename TC, typename TD> 
__forceinline__ __host__ __device__ void cuRotate(TA centerX, TA centerY, TB fcos, TB fsin, TC srcX, TC srcY, TD& dstX, TD& dstY)
{
	dstX = (srcX - centerX)*fcos - (srcY - centerY)*fsin + centerX;
	dstY = (srcX - centerX)*fsin + (srcY - centerY)*fcos + centerY;
}

template<typename T>
struct cuDetectorCoor{
	typedef T DataType;
	__forceinline__ __host__ __device__ cuDetectorCoor(T _du, T _dv, T _nu, T _nv, T _offset_u, T _offset_v)
		:du(_du), dv(_dv), nu(_nu), nv(_nv), offset_u(_offset_u), offset_v(_offset_v),
		centerX((_nu-1.0)/2.0 + _offset_u), centerY((_nv-1.0)/2.0 + _offset_v){
			//centerX = (nu-1.0)/2.0 + offset_u;
			//centerY = (nv-1.0)/2.0 + offset_v;
	}
	template <typename TA, typename TB>  
	__forceinline__ __host__ __device__
		void mm2pixel(TA uMM, TA vMM, TB& u, TB& v) const{
		u = uMM/du;
		v = vMM/dv;
	}
	template <typename TA, typename TB>
	__forceinline__ __host__ __device__ void pixel2mm(TA u, TA v, TB& uMM, TB& vMM) const{
		uMM = u*du;
		vMM = v*dv;
	}
	template <typename TA, typename TB>
	__forceinline__ __host__ __device__ void xy2ij(TA x, TA y, TB& i, TB& j) const{
		i = x + centerX;
		j = centerY - y;
	}
	template <typename TA, typename TB>
	__forceinline__ __host__ __device__ void ij2xy(TA i, TA j, TB& x, TB& y) const{
		x = i - centerX;
		y = centerY - j;
	}
	T du, dv;
	T nu, nv;
	T centerX, centerY;
	T offset_u, offset_v;
};

//struct cuVolumeCoor{
//	__forceinline__ __host__ __device__ cuVolumeCoor(float dx, float dy, float dz, int nx, int ny, int nz){
//		dd = make_float3(dx, dy, dz);
//		nn = make_int3(nx, ny, nz); 
//		center = make_float3((nx-1.0)/2.0f, (ny-1.0)/2.0f, (nz-1.0)/2.0f);
//	}
//	__forceinline__ __host__ __device__ void ijk2xyz(float3 ijk, float3& xyz) const{
//		xyz = center - ijk;
//		xyz.x = -xyz.x;
//		//x = i - centerX;
//		//y = centerY - j;
//		//z = centerZ - k;
//	}
//	__forceinline__ __host__ __device__ void xyz2ijk(float3 xyz, float3& ijk) const{
//		xyz.x = -xyz.x;
//		ijk = center - xyz;
//		//i = x + centerX;
//		//j = centerY - y;
//		//k = centerZ - z;
//	}
//	__forceinline__ __host__ __device__ void pixel2mm(const float3& xyz, float3& xyzMM) const{
//		xyzMM = dd*xyz;
//		//xmm = dx*x;
//		//ymm = dy*y;
//		//zmm = dz*z;
//	}
//	__forceinline__ __host__ __device__ void mm2pixel(const float3& xyzMM, float3& xyz) const{
//		xyz = xyzMM/dd;
//		//x = xmm/dx;
//		//y = ymm/dy;
//		//z = zmm/dz;
//	}
//	float3 dd;
//	int3 nn;
//	float3 center;
//};

template<typename T>
struct cuVolumeCoor{
	typedef T DataType;
	__forceinline__ __host__ __device__ cuVolumeCoor(T dx, T dy, T dz, 
		T nx, T ny, T nz):dx(dx),dy(dy),dz(dz),nx(nx),ny(ny),nz(nz),
		centerX((nx-1.0)/2.0), centerY((ny-1.0)/2.0),centerZ((nz-1.0)/2.0)
	{
			//centerX = (nx-1.0)/2.0; centerY = (ny-1.0)/2.0; centerZ = (nz-1.0)/2.0;
	}
	template <typename TA, typename TB>
	__forceinline__ __host__ __device__ void ijk2xyz(TA i, TA j, TA k, TB& x, TB& y, TB& z) const{
		x = i - centerX;
		y = centerY - j;
		z = centerZ - k;
	}
	template <typename TA, typename TB>
	__forceinline__ __host__ __device__ void xyz2ijk(const TA& x, const TA& y, const TA& z, TB& i, TB& j, TB& k) const{
		i = x + centerX;
		j = centerY - y;
		k = centerZ - z;
	}
	template <typename TA, typename TB>
	__forceinline__ __host__ __device__ void pixel2mm(const TA& x, const TA& y, const TA& z, TB& xmm, TB& ymm, TB& zmm) const{
		xmm = dx*x;
		ymm = dy*y;
		zmm = dz*z;
	}
	template <typename TA, typename TB>
	__forceinline__ __host__ __device__ void mm2pixel(const TA& xmm, const TA& ymm, const TA& zmm, TB& x, TB& y, TB& z) const{
		x = xmm/dx;
		y = ymm/dy;
		z = zmm/dz;
	}
	T dx, dy, dz;
	T nx, ny, nz;
	T centerX, centerY, centerZ;
};


//struct cuVolumeGeo{
//	cuVolumeGeo(int _nx, int _ny, int _nz, float _dx, float _dy, float _dz):
//		nx(_nx), ny(_ny), nz(_nz), dx(_dx), dy(_dy), dz(_dz),tStep(MIN3(_dx, _dy, _dz)){
//			const cuVolumeCoor volCor(this->dx, this->dy, this->dz, this->nx, this->ny, this->nz);
//			float3 box_min_ijk = make_float3(0, this->ny-1, this->nz-1);
//			float3 box_max_ijk = make_float3(this->nx-1, 0, 0);
//			volCor.ijk2xyz(box_min_ijk, boxmin);
//			volCor.ijk2xyz(box_max_ijk, boxmax);
//			volCor.pixel2mm(boxmin, boxmin);
//			volCor.pixel2mm(boxmax, boxmax);
//	}
//	int    nx, ny, nz;
//	float  dx, dy, dz;
//	float3 boxmin, boxmax;
//	float tStep;
//};

struct cuVolumeGeo{
	cuVolumeGeo(int _nx, int _ny, int _nz, float _dx, float _dy, float _dz):
		nx(_nx), ny(_ny), nz(_nz), dx(_dx), dy(_dy), dz(_dz),tStep(MIN3(_dx, _dy, _dz)){
			const cuVolumeCoor<float> volCor(this->dx, this->dy, this->dz, this->nx, this->ny, this->nz);
			Vector3<float> box_min_ijk(0, this->ny-1, this->nz-1);
			Vector3<float> box_max_ijk(this->nx-1, 0, 0);
			volCor.ijk2xyz(box_min_ijk.x, box_min_ijk.y, box_min_ijk.z, boxmin.x, boxmin.y, boxmin.z);
			volCor.ijk2xyz(box_max_ijk.x, box_max_ijk.y, box_max_ijk.z, boxmax.x, boxmax.y, boxmax.z);
			volCor.pixel2mm(boxmin.x, boxmin.y, boxmin.z, boxmin.x, boxmin.y, boxmin.z);
			volCor.pixel2mm(boxmax.x, boxmax.y, boxmax.z, boxmax.x, boxmax.y, boxmax.z);
	}
	int    nx, ny, nz;
	float  dx, dy, dz;
	float3 boxmin, boxmax;
	float tStep;
};

struct cuParameter{
	cuParameter(const Parameter& para){
		du = para.du; dv = para.dv;
		nu = para.nu; nv = para.nv;
		offset_nu = para.offset_nu; offset_nv = para.offset_nv;
		SID = para.SID; SOD = para.SOD;
		//nx = para.nx; ny = para.ny; nz = para.nz;
		//dx = para.dx; dy = para.dy; dz = para.dz;
		
	}

	float du, dv;
	int   nu, nv;
	float  offset_nu, offset_nv;	

    float SID;  
	float SOD;  
	//float3 boxmin, boxmax;
};






template<typename _TpTex> __forceinline__ __device__
float Text3DLinear(_TpTex tex, float3 pos){
#if 1
	//int   ax(x),      ay(y),    az(z);
	//float cx(x-ax), cy(y-ay), cz(z-az);
	//float _cx(1-cx), _cy(1-cy), _cz(1-cz);
	//float tmp[8];
	//#pragma unroll
	//for (int k=0; k<2; k ++){
	//	#pragma unroll
	//	for (int j=0; j<2; j ++){
	//		#pragma unroll
	//		for (int i=0; i<2; i ++){
	//			int _x = ax + i;
	//			int _y = ay + j;
	//			int _z = az + k;
	//			cuPointValue(tmp, 2, 2, 2, i, j, k) = tex3D(tex, _x+0.5, _y+0.5, _z+0.5);			
	//		}
	//	}
	//}
	//float* pDataA = tmp;
	//float* pDataB = &tmp[4];
	//float v0 = (pDataA[0]*(1-cx) + pDataA[1]*cx)*(1-cy) +  (pDataA[2]*(1-cx) + pDataA[2+1]*cx)*cy;
	//float v1 = (pDataB[0]*(1-cx) + pDataB[1]*cx)*(1-cy) +  (pDataB[2]*(1-cx) + pDataB[2+1]*cx)*cy;
	//float val = v0*(1-cz) + v1*cz;
	//return val;

	int3   ax = make_int3(pos.x, pos.y, pos.z);
	float3 cx = pos - make_float3(ax); 
	float3 _cx = make_float3(1, 1, 1) - cx;
	float4 A = make_float4(tex3D(tex, ax.x, ax.y,   ax.z),   tex3D(tex, ax.x+1, ax.y, ax.z),
		                   tex3D(tex, ax.x, ax.y+1, ax.z),   tex3D(tex, ax.x+1, ax.y+1, ax.z));
	float4 B = make_float4(tex3D(tex, ax.x, ax.y,   ax.z+1),   tex3D(tex, ax.x+1, ax.y, ax.z+1),
		                   tex3D(tex, ax.x, ax.y+1, ax.z+1),   tex3D(tex, ax.x+1, ax.y+1, ax.z+1));
	float4 C = A*_cx.z + B*cx.z;
	//float2 D = make_float2(C.x, C.y);
	//float2 E = make_float2(C.z, C.w);
	//float2 F = D*_cx.y + E*cx.y;
	float2 F = make_float2(C.x, C.y)*_cx.y + make_float2(C.z, C.w)*cx.y;;
	float val = F.x*_cx.x + F.y*cx.x;
	//float val = (C.x*_cx.x + C.y*cx.x)*_cx.y +  (C.z*_cx.x + C.w*cx.x)*cx.y;

	return val;
#else
	return tex3D(tex, pos.x+0.5, pos.y+0.5, pos.z+0.5);	
#endif
	
}

bool cudaCTForwardProjection_f32(const ProjectionImageT<float>& proj, const Parameter& para, const Volume& volume, DevArray3D<float>* pDevVolRoi, DevArray3D<float>* pDevVolMask, ImageT<float>& forward_proj, DevData<float>& dev_forward_proj, std::vector<DevData<float>>* vec_dev_forward_proj, ProjectionImageT<float>* pforward_proj_2th = NULL, int proj_index = -1);

bool cudaCTBackprojection_f32(ProjectionImageT<float>& proj, std::vector<DevData<float>>* pvec_dev_proj, DevData<float>& dev_proj, const Parameter& para, VolumeT<float>& vol, DevArray3D<float>& dev_vol, int proj_index = -1);

bool cuSubMul_f32(const DevData<float>& dataA, const DevData<float>& dataB, float scale, const DevData<float>& dataC);


#endif //__cudaIterType_H