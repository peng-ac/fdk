#ifndef __BPLIB_H
#define __BPLIB_H


__device__ __forceinline__ float FMAD(const float a, const float b, const float c) {
#if 0
	return a * b + c;
#else
	float d;
	asm volatile("mad.rz.ftz.f32 %0,%1,%2,%3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
	return d;
#endif
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

__device__ __forceinline__ float dotIJKW(const float4 a, const float4 b) {
	float sum = a.w;
#if 0
	sum = a.x*b.x + sum;
	sum = a.y*b.y + sum;
	sum = a.z*b.z + sum;
	sum = a.w*b.w + sum;
#else
	sum = FMAD(a.x, b.x, sum);
	sum = FMAD(a.y, b.y, sum);
	sum = FMAD(a.z, b.z, sum);
#endif
	return sum;
}

















#endif