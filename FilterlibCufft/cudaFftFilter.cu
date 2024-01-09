//#if 1
#include "cudaBackProjection.cuh"
//#include "../3DCT/type.h"

#include <stdio.h>
#include <math.h>
#include <cufft.h>
#include <assert.h>
#include <vector>
#include "../common/StopWatch.h"



template<typename _Tp0, typename _Tp1>
__global__ void kernelPad(const _Tp0* src, int src_pitch, int src_width, int src_height,
						  _Tp1* dst, int dst_pitch, int dst_width, int dst_height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (!IsInBox(x, y, dst_width, dst_height)) return;

	int pad = (dst_width - src_width)/2;
	if (x < pad){
		int idx = y*src_pitch;
		dst[y*dst_pitch+x] = (src[idx] + src[idx+1] + src[idx+2])/3.0;
		//dst[y*dst_pitch+x] = 0;
	}else if (x >= pad + src_width){
		int idx = y*src_pitch+src_width-1;
		dst[y*dst_pitch+x] = (src[idx] + src[idx-1] + src[idx-2])/3.0;
		//dst[y*dst_pitch+x] = 0;
	}else{
		dst[y*dst_pitch+x] = src[y*src_pitch+x-pad];
	}
}

template<typename _Tp0, typename _Tp1>
__global__ void kernelUnpad(const _Tp0* src, int src_pitch, int src_width, int src_height,
						  _Tp1* dst, int dst_pitch, int dst_width, int dst_height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (!IsInBox(x, y, dst_width, dst_height)) return;

	int pad = (src_width - dst_width)/2;
	dst[dst_pitch*y+x] = src[src_pitch*(y) + pad + x];
}

__global__ void kernelDotScale(cufftComplex* data, int data_pitch, int data_width, int data_height, 
						  const cufftReal* filterReal, int filter_pitch, int filter_width, 
						  float scale)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (!IsInBox(x, y, data_width, data_height)) return;
	//__shared__ float sMem[1024];
	//for (int i=0; i<1024; i ++) sMem[i] = data[i+data_pitch].x*scale;
	cufftComplex* val = data + y*data_pitch + x;
	const cufftReal*    fReal = filterReal + x;
	val->x *= *fReal * scale;
	val->y *= *fReal * scale;
}

template<typename _Tp0, typename _Tp1>
bool FilterRow(const _Tp0* src, _Tp0* dst, int width, int height, const _Tp1* filter, int filter_size){
	bool bRtn = false;
	return bRtn;
}

template<typename _Tp0, typename _Tp1>
bool PadDataRow(const _Tp0* src, int src_pitch, int src_width, int src_height,
						  _Tp1* dst, int dst_pitch, int dst_width, int dst_height)
{
	assert(src_width < dst_width && src_height == dst_height);
	dim3 blocks(32, 32);
	dim3 grids(UpDivide(dst_width, blocks.x), UpDivide(dst_height, blocks.y));
	kernelPad<_Tp0, _Tp1><<<grids, blocks>>>(src, src_pitch, src_width, src_height,
						  dst, dst_pitch, dst_width, dst_height);
	CUDA_CHECK_ERROR;
	return true;
}

inline bool PadDataRow(const gpuData2dReal32& devSrc, gpuData2dReal32& devSrcPad)
{
	return PadDataRow<gpuData2dReal32::DataType, gpuData2dReal32::DataType>(devSrc.Data(), devSrc.DataPitch(), devSrc.width, devSrc.height,
		devSrcPad.Data(), devSrcPad.DataPitch(), devSrcPad.width, devSrcPad.height);
}

inline bool PadDataRow(const gpuData3dReal32& devSrc, gpuData3dReal32& devSrcPad)
{
	return PadDataRow<gpuData2dReal32::DataType, gpuData2dReal32::DataType>(devSrc.Data(), devSrc.DataPitch(), devSrc.width, devSrc.height*devSrc.depth,
		devSrcPad.Data(), devSrcPad.DataPitch(), devSrcPad.width, devSrcPad.height*devSrcPad.depth);
}

template<typename T>
inline bool PadDataRow(const DataRef<T>& devSrc, DataRef<T>& devSrcPad)
{
	return PadDataRow<gpuData2dReal32::DataType, gpuData2dReal32::DataType>(devSrc.data, devSrc.pitch, devSrc.width, devSrc.height*devSrc.depth,
		devSrcPad.data, devSrcPad.pitch, devSrcPad.width, devSrcPad.height*devSrcPad.depth);
}

template<typename T>
inline bool UnpadDataRow(const DataRef<T>& devSrc, DataRef<T>& devDst)
{
	dim3 blocks(32, 32);
	dim3 grids(UpDivide(devDst.width, blocks.x), UpDivide(devDst.height*devDst.depth, blocks.y));
	kernelUnpad<float, float><<<grids, blocks>>>(devSrc.data, devSrc.pitch, devSrc.width, devSrc.height*devSrc.depth,
		devDst.data, devDst.pitch, devDst.width, devDst.height*devDst.depth);
	CUDA_CHECK_ERROR;
	return true;
}

inline bool UnpadDataRowReal32(const gpuData2dReal32& devSrc, gpuData2dReal32& devDst)
{
	auto src = devSrc.GetDataRef();
	auto dst = devDst.GetDataRef();
	return UnpadDataRow<gpuData2dReal32::DataType>(src, dst);
}

inline bool ComplexDotScale(cufftComplex* data, cufftReal* filterReal, float scale, int pitch, int width, int height)
{
	assert(pitch == width);
	dim3 blocks(32, 32);
	dim3 grids(UpDivide(width, blocks.x), UpDivide(height, blocks.y));
	kernelDotScale<<<grids, blocks>>>(data, pitch, width, height, filterReal, pitch, width, scale);
	CUDA_CHECK_ERROR;
	return true;
}

float dot(float2 a){
	return sqrt(a.x*a.x + a.y*a.y);
}

bool FilterRow(const float* src, float* dst, int width, int height, const float* filter, int filter_size){
	bool bRtn = false;
	DWORD dwTm = 0;
	
	dwTm = timeGetTime();	
	gpuData2dReal32 devSrc(width, height);
	gpuData2dReal32 devSrcPad(filter_size, height);
	gpuData1dReal32 devFilter(filter_size);		

	devFilter.CopyHostToDevice(filter, filter_size);

	//devFilter.Display("devFilter");
	//WaitKey();
	devSrc.CopyHostToDevice(src, width, height);
	//devSrc.Display("devSrc");
	//WaitKey();
	dwTm = timeGetTime() - dwTm;	

	gpuData2dReal32 devDstPad(filter_size, height);
	gpuData2dReal32 devDst(width, height);	
	gpuData<cufftComplex, 2> devDataComplex(filter_size, height);

	PadDataRow(devSrc, devSrcPad);
	//devSrcPad.Display("devSrcPad");
	//WaitKey();

	assert(devSrcPad.DataPitch() == devSrcPad.width);
	assert(devSrcPad.DataPitch() == filter_size);

	std::vector<cufftHandle> vecPlanFFT(height+1);
	std::vector<cufftHandle> vecPlanIFFT(height+1);
	{
#if 0
		//cudaMemcpy(devDataComplex.Data(), devSrcPad.Data(), devSrcPad.pitch*devSrcPad.height, cudaMemcpyDeviceToDevice);
		for (int i=0; i<height; i ++){
			cufftHandle& plan = vecPlanFFT[i];
			CUFFT_CHECK_ERROR(cufftPlan1d(&plan, filter_size, CUFFT_R2C, 1));
			CUFFT_CHECK_ERROR(cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_FFTW_ALL));
			CUFFT_CHECK_ERROR(cufftExecR2C(plan, devSrcPad.Data() + i * devSrcPad.DataPitch(), devDataComplex.Data() + i*devDataComplex.DataPitch()));

		}
#else
		{
			cufftHandle plan = NULL;
			const int rank = 1;
			int n[rank] = {filter_size};
			int inembed[rank] = {filter_size};
			const int istride = 1;
			const int idist = filter_size;
			int onembed[rank] = {filter_size};
			const int ostride = 1;
			const int odist = filter_size; 
			const cufftType type = CUFFT_R2C; 
			const int batch = height;
			CUFFT_CHECK_ERROR(cufftPlanMany(
				&plan, rank, n, 
				inembed, istride, idist, 
				onembed, ostride, odist, 
				type, batch));
			CUFFT_CHECK_ERROR(cufftExecR2C(plan, devSrcPad.Data(), devDataComplex.Data()));
		}
#endif
		//cudaThreadSynchronize();
		CUDA_CHECK_ERROR;
	}
	
	//std::vector<float> tmp_pad(devDstPad.DataPitch()*devDstPad.height);

	//devDataComplex.Display("devDataComplex", dot);
	//WaitKey();
	float scale = 1.0/devFilter.width;
	ComplexDotScale(devDataComplex.Data(), devFilter.Data(), scale, devDataComplex.DataPitch(), devDataComplex.width, devDataComplex.height);	
		
	//ifft
	{
#if 0
		for (int i=0; i<height; i ++)
		{
			cufftHandle& plan = vecPlanIFFT[i];
			CUFFT_CHECK_ERROR(cufftPlan1d(&plan, filter_size, CUFFT_C2R, height));
			CUFFT_CHECK_ERROR(cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_FFTW_ALL));
			CUFFT_CHECK_ERROR(cufftExecC2R(plan, (cufftComplex*)devDataComplex.Data() + i*devDataComplex.DataPitch(), devDstPad.Data() + i*devDstPad.DataPitch()));
			//CUFFT_CHECK_ERROR(cufftExecC2R(plan, devDataComplex.Data(), devDstPad.Data()));
		}
#else	
		{
			cufftHandle plan = NULL;
			const int rank = 1;
			int n[rank] = {filter_size};
			int inembed[rank] = {filter_size};
			const int istride = 1;
			const int idist = filter_size;
			int onembed[rank] = {filter_size};
			const int ostride = 1;
			const int odist = filter_size; 
			const cufftType type = CUFFT_C2R; 
			const int batch = height;
			CUFFT_CHECK_ERROR(cufftPlanMany(
				&plan, rank, n, 
				inembed, istride, idist, 
				onembed, ostride, odist, 
				type, batch));

			CUFFT_CHECK_ERROR(cufftExecC2R(plan, devDataComplex.Data(), devDstPad.Data()));
		}	
#endif
		cudaThreadSynchronize();
		CUDA_CHECK_ERROR;

		devDstPad.Display("devDstPad");
		//WaitKey();
	}

	//for (int i=0; i<height; i ++) if (vecPlanFFT[i]) cufftDestroy(vecPlanFFT[i]);
	//for (int i=0; i<height; i ++) if (vecPlanIFFT[i]) cufftDestroy(vecPlanIFFT[i]);
	
	UnpadDataRowReal32(devDstPad, devDst);
		

	//devDst.CopyDeviceToHost(&tmp_pad[0], devDst.width, devDst.height);
	////compare
	//for (int i=0; i<devDst.height; i ++){
	//	float dif = 0;
	//	for (int j=0; j<devDst.width; j ++){
	//		dif += fabs(tmp_pad[i*devDst.width + j] - img_dst(j, i));
	//	}
	//	dif /= devDst.width;
	//	printf("%d : dif = %f\n", i, dif);
	//}

	std::cout<<"time = "<<dwTm<<std::endl;
	devDst.Display("devDst");
	WaitKey();

	return bRtn;
}

template<typename _Tp0, typename _Tp1> 
__global__ void kernelCorrect(const _Tp0* src, _Tp0* dst, int pitch, int width, int height, int depth, const _Tp1* table, int table_pitch)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (IsInBox(x, y, z, width, height, depth) == false) return;

	const int offset_0 = z*pitch*height + pitch*y + x;
	const int offset_1 = y*table_pitch + x;
	dst[offset_0] = src[offset_0]*table[offset_1];
}

template<typename T> inline
bool CorrectImage(const DataRef<T>& src, const DataRef<T>& table, DataRef<T>& dst)
{
	VERIFY_TRUE(src.pitch  == dst.pitch);
	VERIFY_TRUE(src.width  == dst.width);
	VERIFY_TRUE(src.height == dst.height);
	VERIFY_TRUE(src.depth  <= dst.depth);
	VERIFY_TRUE(table.Is2D());
	dim3 blocks(32, 4, 4);
	dim3 grids(UpDivide(src.width, blocks.x), UpDivide(src.height, blocks.y), UpDivide(src.depth, blocks.z));
	kernelCorrect<<<grids, blocks>>>(src.data, dst.data, src.pitch, src.width, src.height, src.depth, table.data, table.pitch);
	CUDA_CHECK_ERROR;
	return true;
}

template<typename T> inline
bool FilterRow(const DataRef<T>& src, DataRef<T>& dst, const DataRef<T>& filter)
{
	bool bRtn = false;
	VERIFY_TRUE(src.width  == dst.width);
	VERIFY_TRUE(src.height == dst.height);
	VERIFY_TRUE(src.depth  <= dst.depth);
	VERIFY_TRUE(filter.height == 1);
	const int filter_size = filter.width;
	const int width  = src.width;
	const int height = src.height;
	const int depth  = src.depth;
	
	gpuData3dReal32_2 _devDataComplex(filter_size, height, depth);
	gpuData3dReal32 _devPad(filter_size, height, depth);
	gpuData3dReal32_2::DataRefType devDataComplex = _devDataComplex.GetDataRef();
	gpuData3dReal32::DataRefType  devPad = _devPad.GetDataRef();

	PadDataRow(src, devPad);
	//for (int i=0; i<depth; i ++){ 
	//	devPad.Display("devSrcPad", i);	
	//}
	//WaitKey();
	{
		cufftHandle plan = NULL;
		const int rank = 1;
		int n[rank] = {filter_size};
		int inembed[rank] = {filter_size};
		const int istride = 1;
		const int idist = filter_size;
		int onembed[rank] = {filter_size};
		const int ostride = 1;
		const int odist = filter_size; 
		const int batch = height*depth;
		const cufftType type = CUFFT_R2C; 
		{
			//FFT
			CUFFT_CHECK_ERROR(cufftPlanMany(
				&plan, rank, n, 
				inembed, istride, idist, 
				onembed, ostride, odist, 
				type, batch));
			CUFFT_CHECK_ERROR(cufftExecR2C(plan, devPad.data, devDataComplex.data));
			CUDA_CHECK_ERROR;
			::cufftDestroy(plan);
		}
		{
			//Complex Dot product & scale fft
			float scale = 1.0/filter_size;
			ComplexDotScale(devDataComplex.data, filter.data, scale, devDataComplex.pitch, devDataComplex.width, devDataComplex.height*devDataComplex.depth);
		}
		{
			//IFFT
			const cufftType type = CUFFT_C2R; 
			CUFFT_CHECK_ERROR(cufftPlanMany(
				&plan, rank, n, 
				inembed, istride, idist, 
				onembed, ostride, odist, 
				type, batch));
			CUFFT_CHECK_ERROR(cufftExecC2R(plan, devDataComplex.data, devPad.data));
			::cufftDestroy(plan);
		}
		//for (int i=0; i<depth; i ++){ 
		//	_devPad.Display("devSrcPad", i);		
		//}
		//WaitKey();
	}
	UnpadDataRow<T>(devPad, dst);
	//dst.Display("dst0", 0);
	//dst.Display("dst1", 1);

	return bRtn;
}

inline bool FilterRow(const gpuData3dReal32& src, gpuData3dReal32& dst, const gpuData1dReal32& filter)
{
	const gpuData3dReal32::DataRefType dataSrc = src.GetDataRef();
	const gpuData1dReal32::DataRefType dataFilter = filter.GetDataRef();
	VERIFY_TRUE(dataFilter.Is1D());
	gpuData3dReal32::DataRefType dataDst = dst.GetDataRef();

	return FilterRow<gpuData3dReal32::DataType>(dataSrc, dataDst, dataFilter);
}

bool FilterRow_Real32(const DataRef<float>& src, DataRef<float>& dst, const DataRef<float>& filter){
	return FilterRow(src, dst, filter);
}

bool CorrectImage_Real32(const DataRef<float>& src, const DataRef<float>& table, DataRef<float>& dst){
	DataRef<float> _src = src;
	const DataRef<float> _table = table;
	DataRef<float> _dst = dst;

	return CorrectImage(_src, _table, _dst);
}

float FilterProjImage(const Parameter& para, const ProjectionImage& proj_img, int DEPTH=32)
{
	extern bool FilterRow_Real32(const DataRef<float>& src, DataRef<float>& dst, const DataRef<float>& filter);
	extern bool CorrectImage_Real32(const DataRef<float>& src, const DataRef<float>& table, DataRef<float>& dst);

	StopWatchWin watch;
	//printf("FilterProjImage");
	DWORD dwTm = timeGetTime();
	//const int DEPTH = 32;
	std::cout << "FilterProjImage, BATCH Count = " << DEPTH << std::endl;
	DWORD dwTotalTm = timeGetTime() - dwTm;
	watch.start();
	const ImageT<float>& cos_table = para.cos_image_table.cos_sigma;
	gpuData2dReal32 devTable(cos_table.width, cos_table.height);
	devTable.CopyHostToDevice(cos_table.buffer, cos_table.width, cos_table.height);
	gpuData3dReal32 devSrc(proj_img.width, proj_img.height, DEPTH);
	gpuData1dReal32 devFilter(para.fft_kernel_real_f32.size());
	gpuData3dReal32 devDst(proj_img.width, proj_img.height, DEPTH);
	devFilter.CopyHostToDevice(&para.fft_kernel_real_f32[0], para.fft_kernel_real_f32.size());
	watch.stop();
	dwTotalTm = 0; // timeGetTime() - dwTm;

	for (int i = 0; i < proj_img.GetCount(); i += DEPTH) {
		dwTm = timeGetTime();
		ProjectionImage::FrameType frame;
		proj_img.GetFrame(frame, i);
		int count = MIN(DEPTH, proj_img.GetCount() - i);
		//devSrc.CopyHostToDevice(proj_img.Buffer(), proj_img.width, proj_img.height, depth);
		devSrc.CopyHostToDevice(frame.buffer, proj_img.width, proj_img.height, count);

		auto a = devSrc.GetDataRef();
		auto b = devTable.GetDataRef();
		auto c = devSrc.GetDataRef();

		CorrectImage_Real32(a, b, c);
		const gpuData3dReal32::DataRefType srcRef = devSrc.GetDataRef();
		gpuData3dReal32::DataRefType dstRef = devDst.GetDataRef();
		dstRef.depth = min(dstRef.depth, srcRef.depth);
		FilterRow_Real32(srcRef, dstRef, devFilter.GetDataRef());
		dstRef.MemcpyDeviceToHost(frame.buffer, frame.width, frame.height, count);
		dwTm = timeGetTime() - dwTm;
		dwTotalTm += dwTm;
		//std::cout<<i<<": "<< depth<<" Filter Row Time = "<<dwTm<<std::endl;
		//ClearCurrentLine();
		printf("FilterProjImage:%02d, count=%d, tm = %d, total_time=%f s\n", i, DEPTH, dwTm, dwTotalTm / 1000.);
		//for (int i=0; i<depth; i ++){ 
		//	devDst.Display("devDst", i);	
		//	WaitKey(30);
		//}	
	}
	//printf("\n");
	//for (int i=0; i<proj_img.GetCount(); i ++){ 
	//	ProjectionImage::FrameType frame;
	//	proj_img.GetFrame(frame, i);
	//	frame.display("FilterProjImage");
	//	WaitKey(10);
	//}	
	return dwTotalTm;
}

void FilterImage(const Parameter& para, const ProjectionImage& proj_img)
{
	DWORD dwTm = timeGetTime();	
	int depth = 32;
	gpuData3dReal32 devSrc(proj_img.width, proj_img.height, depth);
	gpuData1dReal32 devFilter(para.fft_kernel_real_f32.size());
	gpuData3dReal32 devDst(proj_img.width, proj_img.height, depth);

	devFilter.CopyHostToDevice(&para.fft_kernel_real_f32[0], para.fft_kernel_real_f32.size());

	for (int i=0; i<proj_img.GetCount(); i += depth){
		ProjectionImage::FrameType frame;
		proj_img.GetFrame(frame, i);
		//devSrc.CopyHostToDevice(proj_img.Buffer(), proj_img.width, proj_img.height, depth);
		devSrc.CopyHostToDevice(frame.buffer, proj_img.width, proj_img.height, MIN(depth, proj_img.GetCount() - i));
		dwTm = timeGetTime();
		const gpuData3dReal32::DataRefType srcRef = devSrc.GetDataRef();
		gpuData3dReal32::DataRefType dstRef = devDst.GetDataRef();
		dstRef.depth = min(dstRef.depth, srcRef.depth);
		FilterRow(srcRef, dstRef, devFilter.GetDataRef());
		dwTm = timeGetTime() - dwTm;
		std::cout<<i<<": "<< depth<<" Filter Row Time = "<<dwTm<<std::endl;
		for (int i=0; i<depth; i ++){ 
			devDst.Display("devDst", i);	
			WaitKey(30);
		}	
	}

}

void TestFilter()
{
	DISPLAY_FUNCTION;
	bool bRtn;
	Parameter para;
	ProjectionImage proj;
	char szPath[1024] = "/media/peng/DATADRIVE2/CtData/proj-img/";
	bRtn = proj.LoadFolder<float>(szPath, para.nu, para.nv, para.nProj, 1);
	FrameT<float> frame0, frame1, frame2;
	proj.GetFrame(frame0, 1);
	proj.GetFrame(frame1, 180);
	memcpy(frame0.buffer, frame1.buffer, sizeof(float)*frame0.width*frame0.height);


	VERIFY_TRUE(bRtn);
	FilterImage(para, proj);
	return;

	FrameT<float> frame;
	bRtn = proj.GetFrame(frame, 0);
	VERIFY_TRUE(bRtn);
	ImageT<float> img_dst(frame.width, frame.height);
	//filterImageRow(para.fft_kernel_real, frame.data, img_dst.buffer, frame.width, frame.height);
	//img_dst.display("img_dst");
	//WaitKey();

	FilterRow(frame.buffer, img_dst.buffer, frame.width, frame.height, &para.fft_kernel_real_f32[0], para.fft_kernel_real_f32.size());
}

void TestCorrect()
{
	bool bRtn;
	Parameter para;
	para.Display();
	ProjectionImage proj;
	std::vector<gpuData3dReal32::DataType> vecData(proj.width*proj.height+1);
	for (int i=0; i<vecData.size(); i ++) 
		vecData[i] = 2;
	char szPath[1024] = "/media/peng/DATADRIVE2/CtData/proj-img/";
	proj.MallocBuffer(para.nu, para.nv, 32, para.step);
	VERIFY_TRUE(proj.IsValid());
	for (int i=0; i<proj.width*proj.height*proj.GetCount(); i ++) 
		proj.Buffer()[i] = i;

	gpuData3dReal32 devProjImg(proj.width, proj.height, proj.GetCount());
	devProjImg.CopyHostToDevice(proj.Buffer(), proj.width, proj.height, proj.GetCount());

	gpuData2dReal32 devTable(proj.width, proj.height);
	devTable.CopyHostToDevice(&vecData[0], proj.width, proj.height);

	auto src = devProjImg.GetDataRef();
	auto table = devTable.GetDataRef();
	auto dst =  devProjImg.GetDataRef();
	CorrectImage(src, table, dst);

	std::vector<gpuData3dReal32::DataType> vecDataDst(proj.width*proj.height*proj.GetCount()+1);
	devProjImg.CopyDeviceToHost(&vecDataDst[0], proj.width, proj.height, proj.GetCount());

	return;
}


int mainFilter(int argc, char** argv)
{
	TestCorrect();
	//TestFilter();
	return 0;
}


//#endif
