#ifndef __IPPLIB_H
#define __IPPLIB_H

#include <stdio.h>
#include <assert.h>
typedef unsigned char uchar;
typedef unsigned short ushort;

#ifdef _WIN32
#pragma comment(lib, "ipps.lib")
#pragma comment(lib, "ippi.lib")
#pragma comment(lib, "ippcore.lib")
#pragma comment(lib, "ippvm.lib")
#endif // !

#include "ipps.h"
#include "ippi.h"
template<typename T> inline IppStatus ippsCopy(const T* pSrc, T* pDst, int len){
	memcpy(pDst, pSrc, sizeof(pSrc[0])*len);
	return ippStsNoErr;
}

template<> inline IppStatus ippsCopy<float>(const float* pSrc, float* pDst, int len){
	return ippsCopy_32f(pSrc, pDst, len);
}

template<> inline IppStatus ippsCopy<Ipp16s>(const Ipp16s* pSrc, Ipp16s* pDst, int len) {
	return ippsCopy_16s(pSrc, pDst, len);
}

template<typename T> inline IppStatus ippsCopy(const T* pSrc, int srcStep, T* pDst, int dstStep, int roi_width, int roi_height) {
	assert(0);
	return ippStsNoErr;
}
template<> inline IppStatus ippsCopy<float>(const float* pSrc, int srcStep, float* pDst, int dstStep, int roi_width, int roi_height) {
	const IppiSize roiSize = { roi_width, roi_height };
	return ippiCopy_32f_C1R(pSrc, srcStep*sizeof(pSrc[0]), pDst, dstStep*sizeof(pDst[0]), roiSize);
}

template<typename T> inline IppStatus ippiTranspose(const T* pSrc, int srcStep, T* pDst, int dstStep, int src_roi_width, int src_roi_height){
	assert(0);
	return ~ippStsNoErr;
}

template<> inline IppStatus ippiTranspose<float>(const float* pSrc, int srcStep, float* pDst, int dstStep, int src_roi_width, int src_roi_height){
	const IppiSize srcRoi = {src_roi_width, src_roi_height};
	return ippiTranspose_32f_C1R(pSrc, srcStep*sizeof(pSrc[0]), pDst, dstStep*sizeof(pDst[0]), srcRoi);
}

template<> inline IppStatus ippiTranspose<short>(const short* pSrc, int srcStep, short* pDst, int dstStep, int src_roi_width, int src_roi_height){
	const IppiSize srcRoi = {src_roi_width, src_roi_height};
	return ippiTranspose_16s_C1R(pSrc, srcStep*sizeof(pSrc[0]), pDst, dstStep*sizeof(pDst[0]), srcRoi);
}

template<> inline IppStatus ippiTranspose<ushort>(const ushort* pSrc, int srcStep, ushort* pDst, int dstStep, int src_roi_width, int src_roi_height){
	const IppiSize srcRoi = {src_roi_width, src_roi_height};
	return ippiTranspose_16u_C1R(pSrc, srcStep*sizeof(pSrc[0]), pDst, dstStep*sizeof(pDst[0]), srcRoi);
}

template<> inline IppStatus ippiTranspose<uchar>(const uchar* pSrc, int srcStep, uchar* pDst, int dstStep, int src_roi_width, int src_roi_height){
	const IppiSize srcRoi = {src_roi_width, src_roi_height};
	return ippiTranspose_8u_C1R(pSrc, srcStep, pDst, dstStep, srcRoi);
}

template<typename T> inline IppStatus ippiMul(const T* pSrc1, int src1Step, const T* pSrc2, int src2Step, T* pDst, int dstStep, int src_roi_width, int src_roi_height){
	assert(0);
}

template<> inline IppStatus ippiMul<float>(const float* pSrc1, int src1Step, const float* pSrc2, int src2Step, float* pDst, int dstStep, int src_roi_width, int src_roi_height){
	const IppiSize srcRoi = {src_roi_width, src_roi_height};
	return ippiMul_32f_C1R(pSrc1, src1Step, pSrc2, src2Step, pDst, dstStep, srcRoi);
}

class ippsFFT{
public:
	ippsFFT(int _order, int _flag = IPP_FFT_DIV_INV_BY_N, IppHintAlgorithm _hint = ippAlgHintFast)
		:order(_order), flag(_flag), hint(_hint){
			Initialize(order, flag, hint);
	}
	virtual ~ippsFFT(){
		if (pFFTSpecBuf) ippsFree(pFFTSpecBuf);
		if (pFFTInitBuf) ippsFree(pFFTInitBuf);
		if (pFFTWorkBuf) ippsFree(pFFTWorkBuf);
	}
	ippsFFT& operator=(const ippsFFT& obj){
		bool bRtn = Initialize(obj.order, obj.flag, obj.hint);
		assert(bRtn);
		return *this;
	}
	inline bool Initialize(int _order, int flag = IPP_FFT_DIV_INV_BY_N, IppHintAlgorithm hint = ippAlgHintAccurate){
		IppStatus status0 = ippsFFTGetSize_C_32f(order, flag, hint, &sizeFFTSpec, &sizeFFTInitBuf, &sizeFFTWorkBuf);
		assert(status0 == ippStsNoErr);
		pFFTSpecBuf = ippsMalloc_8u(sizeFFTSpec);
		pFFTInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
		pFFTWorkBuf = ippsMalloc_8u(sizeFFTWorkBuf);
		IppStatus status1 = ippsFFTInit_C_32f(&pFFTSpec, order, flag, hint, pFFTSpecBuf, pFFTInitBuf);
		assert(status1 == ippStsNoErr);
		return (status0 == ippStsNoErr && status1 == ippStsNoErr);
	}
	//this function is not thread-safe, due to the Working buffer
	inline int FFTFwd_CToC_32f_I(Ipp32f* pSrcDstRe, Ipp32f* pSrcDstIm){
						 return ippsFFTFwd_CToC_32f_I(pSrcDstRe, pSrcDstIm, pFFTSpec, pFFTWorkBuf);
	}
	//this function is not thread-safe, due to the Working buffer
	inline int FFTInv_CToC_32f_I(Ipp32f* pSrcDstRe, Ipp32f* pSrcDstIm){
						 return ippsFFTInv_CToC_32f_I(pSrcDstRe, pSrcDstIm, pFFTSpec, pFFTWorkBuf);
	}
	IppsFFTSpec_C_32f *pFFTSpec;
	Ipp8u  *pFFTSpecBuf, *pFFTInitBuf, *pFFTWorkBuf;
	int sizeFFTSpec,sizeFFTInitBuf,sizeFFTWorkBuf;
	int              order;
	int              flag;
	IppHintAlgorithm hint;
};



















#endif //__IPPLIB_H