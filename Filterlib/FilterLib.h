#ifndef __FILTER_LIB_H
#define __FILTER_LIB_H

#include "../common/type.h"
#include "../IppLib/IppLib.h"
#include "../common/StopWatch.h"

inline bool FilterCosCorrect(const FrameT<float>& src, const FrameT<float>& cos_tbl, const FrameT<float>& dst){
	VERIFY_TRUE(src.width == cos_tbl.width && src.height == cos_tbl.height && src.width == dst.width && src.height == dst.height);
	const int size = src.width*src.height;
	ippiMul(src.buffer, size, cos_tbl.buffer, size, dst.buffer, size, size, 1);
	return true;
}

inline bool FilterImageRow(const FrameT<float>& src, FrameT<float>& dst_real, FrameT<float>& dst_imag, const std::vector<float>& vecFilter, ippsFFT& fft)
{
	VERIFY_TRUE(src.buffer && dst_real.buffer && dst_imag.buffer);
	VERIFY_TRUE(src.height == dst_imag.height && dst_real.height == dst_imag.height);
	VERIFY_TRUE(dst_real.width == pow(2, fft.order));
	VERIFY_TRUE(dst_real.width == vecFilter.size());
	const int offset = (dst_real.width - src.width)>>1;
	const float* filter = &vecFilter[0];
	const int size = src.width*src.height;
	//dst_real.Zero();
	//dst_imag.Zero();
	ippsZero_32f(dst_real.buffer, size);
	ippsZero_32f(dst_imag.buffer, size);
	for (int i=0; i<src.height; i ++){
		{
			const float* psrc = src.buffer + i*src.width;
			float* pdst = dst_real.buffer + i*dst_real.width + offset;
			ippsCopy_32f(psrc, pdst, src.width);
		}
		{
			const float* psrc = src.buffer + i*src.width;
			float* pdst = dst_real.buffer + i*dst_real.width;
			{
				float val = (psrc[0] + psrc[1] + psrc[2] + psrc[3])/4;
				ippsSet_32f(val, pdst, offset);
			}
			psrc += src.width;
			pdst += src.width + offset;
			{
				float val = (psrc[-1] + psrc[-2] + psrc[-3] + psrc[-4])/4;
				ippsSet_32f(val, pdst, offset);
			}
		}
		{
			float* pdst_real = dst_real.buffer + i*dst_real.width;
			float* pdst_imag = dst_imag.buffer + i*dst_real.width;
			fft.FFTFwd_CToC_32f_I(pdst_real, pdst_imag);

			ippiMul(pdst_real, dst_real.width, filter, dst_real.width, pdst_real, dst_real.width, dst_real.width, 1);
			ippiMul(pdst_imag, dst_real.width, filter, dst_real.width, pdst_imag, dst_real.width, dst_real.width, 1);

			fft.FFTInv_CToC_32f_I(pdst_real, pdst_imag);
		}	
	}
//	const int nSymitric = 32;
//#if 0
//	for (int i=0; i<src.height; i ++){
//		const float* psrc = src.buffer + i*src.width;
//		float* pdst = dst_real.buffer + i*dst_real.width + offset;
//		for (int j=0; j<nSymitric; j ++){
//			pdst[-1-j] = psrc[j];
//		}
//		psrc += src.width;
//		pdst += src.width;
//		for (int j=0; j<nSymitric; j ++){
//			pdst[j] = psrc[-1-j];
//		}
//	}
//#else
//#if 0
//	for (int i=0; i<src.height; i ++){
//		const float* psrc = src.buffer + i*src.width;
//		float* pdst = dst_real.buffer + i*dst_real.width;
//		{
//			float val = (psrc[0] + psrc[1] + psrc[2] + psrc[3])/4;
//			ippsSet_32f(val, pdst, offset);
//		}
//		psrc += src.width;
//		pdst += src.width + offset;
//		{
//			float val = (psrc[-1] + psrc[-2] + psrc[-3] + psrc[-4])/4;
//			ippsSet_32f(val, pdst, offset);
//		}
//	}
//#endif
//#endif
//	//dst_real.display("dst_real");
//	for (int i=0; i<src.height; i ++){
//		float* pdst_real = dst_real.buffer + i*dst_real.width;
//		float* pdst_imag = dst_imag.buffer + i*dst_real.width;
//		fft.FFTFwd_CToC_32f_I(pdst_real, pdst_imag);
//
//		ippiMul(pdst_real, dst_real.width, filter, dst_real.width, pdst_real, dst_real.width, dst_real.width, 1);
//		ippiMul(pdst_imag, dst_real.width, filter, dst_real.width, pdst_imag, dst_real.width, dst_real.width, 1);
//
//		fft.FFTInv_CToC_32f_I(pdst_real, pdst_imag);
//	}
//
	return true;
}

template<int threads>
struct FilterImageRowMulti{
	FilterImageRowMulti(int order) {
		vec_fft.resize(threads);
		for (int i = 0; i < vec_fft.size(); i++){
			vec_fft[i] = new ippsFFT(order);		
		}
	}
	virtual~FilterImageRowMulti() {	
		for (int i = 0; i < vec_fft.size(); i++) {
			if (vec_fft[i]) delete vec_fft[i];	
		}
	}
	inline void operator()(const FrameT<float>& src, FrameT<float>& dst_real, FrameT<float>& dst_imag, const std::vector<float>& vecFilter) {
		int hstep = src.height / threads;
		//DISPLAY_FUNCTION;
		//return;
		if (threads > 1)
		{
			#pragma omp parallel for num_threads(threads)
			for (int i = 0; i < threads; i++)
			{
				const FrameT<float> _src(src.buffer + src.width*hstep*i, src.width, hstep, i);
				FrameT<float> _dst_real(dst_real.buffer + dst_real.width*hstep*i, dst_real.width, hstep, i);
				FrameT<float> _dst_imag(dst_imag.buffer + dst_imag.width*hstep*i, dst_imag.width, hstep, i);
				VERIFY_TRUE(FilterImageRow(_src, _dst_real, _dst_imag, vecFilter, *vec_fft[i]));
			}			
		}else{
			for (int i = 0; i < threads; i++)
			{
				const FrameT<float> _src(src.buffer + src.width*hstep*i, src.width, hstep, i);
				FrameT<float> _dst_real(dst_real.buffer + dst_real.width*hstep*i, dst_real.width, hstep, i);
				FrameT<float> _dst_imag(dst_imag.buffer + dst_imag.width*hstep*i, dst_imag.width, hstep, i);
				VERIFY_TRUE(FilterImageRow(_src, _dst_real, _dst_imag, vecFilter, *vec_fft[i]));
			}
		}

	}
	std::vector<ippsFFT*> vec_fft;
};


struct FilterEngine {
	enum{
		THREAD_COUNT = 1,
	};
	FilterEngine(FrameT<float> _cos_image_table, const std::vector<float>& _fft_kernel_real_f32)
		: fft_kernel_real_f32(_fft_kernel_real_f32)
		, cos_image_table(_cos_image_table) 
	{
#if 1
		dst_real.MallocBuffer(_fft_kernel_real_f32.size(), _cos_image_table.height);
		dst_imag.MallocBuffer(_fft_kernel_real_f32.size(), _cos_image_table.height);	
		rect = Rect32i((_fft_kernel_real_f32.size() - _cos_image_table.width) / 2, 0, _cos_image_table.width, _cos_image_table.height);
		order = log(_fft_kernel_real_f32.size()) / log(2) + 0.5;
		pFilterImageRow = std::make_shared<FilterImageRowMulti<THREAD_COUNT>>(order);
		VERIFY_TRUE(pFilterImageRow);
#endif
	}
	virtual~FilterEngine() {
	}
	FilterEngine& Filtering(FrameT<float>& src, FrameT<float>& dst, bool transpose = false){
		//DISPLAY_FUNCTION;
		//printf("src.width=%d,cos_image_table.width=%d\n", src.width, cos_image_table.width);
		VERIFY_TRUE(src.width == cos_image_table.width && src.height == cos_image_table.height);
		if (transpose){
			VERIFY_TRUE(dst.height == cos_image_table.width && dst.width == cos_image_table.height);
		}else{
			VERIFY_TRUE(dst.width == cos_image_table.width && dst.height == cos_image_table.height);
		}
		auto _dst_real = dst_real.GetFrame();
		auto _dst_imag = dst_imag.GetFrame();
		{	
			//StopWatchWin w0, w1, w2;
			//w0.start();
			FilterCosCorrect(src, cos_image_table, src);	
			//w0.stop();
			//w1.start();
			if (pFilterImageRow)
				(*pFilterImageRow)(src, _dst_real, _dst_imag, fft_kernel_real_f32);	
			//w1.stop();
			//w2.start();
			if (!transpose) ippsCopy(dst_real.buffer + rect.x, dst_real.width, dst.buffer, dst.width, dst.width, dst.height);
			else            ippiTranspose(dst_real.buffer + rect.x, dst_real.width, dst.buffer, dst.width, dst.height, dst.width); 
			//w2.stop();
			//printf("filter-lib, CosCorrect=%f ms,Flting=%f ms, trans=%f ms\n", w0.getAverageTime(), w1.getAverageTime(), w2.getAverageTime());
		}
	
		//WriteToFile<float, float>(StringFormat("%s/img%04d.raw", para.str_working_folder.c_str(), i), dst.buffer, dst.width*dst.height);	
		return *this;
	}
	Rect32i rect;
	ImageT<float> dst_real, dst_imag;
	const std::vector<float>& fft_kernel_real_f32;
	FrameT<float> cos_image_table;
	std::shared_ptr<FilterImageRowMulti<THREAD_COUNT>> pFilterImageRow;
	int order;
};





#endif //__FILTER_LIB_H