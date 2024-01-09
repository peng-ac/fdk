#include <iostream>
#include "../common/type.h"
#include <stdio.h>
//#include <intrin.h>

using namespace std;

void TransposeC(const float* src, int src_width, int src_height, float* dst, int dst_width, int dst_height)
{
	float* p0        = dst;
	for (int i = 0; i < dst_height; i++) {
		for (int j = 0; j < dst_width; j++, p0) {
			*p0 = src[j*src_width + i];
		}
	}
}

void TransposeSSE(const float* src, int src_width, int src_height, float* dst, int dst_width, int dst_height)
{
	__m128 mm0, mm1, mm2, mm3;

	const int src_width_4 = src_width << 2;
	for (int i = 0; i < dst_height; i += 4) {
		float* p0 = dst + dst_width*i;
		float* p1 = p0 + dst_width;
		float* p2 = p1 + dst_width;
		float* p3 = p2 + dst_width;

		const float* ps0 = src + i;
		const float* ps1 = ps0 + src_width;
		const float* ps2 = ps1 + src_width;
		const float* ps3 = ps2 + src_width;
		for (int j = 0; j < dst_width; j += 4, p0 += 4, p1 += 4, p2 += 4, p3 += 4, ps0 += src_width_4, ps1 += src_width_4, ps2 += src_width_4, ps3 += src_width_4) {
			mm0 = _mm_load_ps(ps0);
			mm1 = _mm_load_ps(ps1);
			mm2 = _mm_load_ps(ps2);
			mm3 = _mm_load_ps(ps3);
			_MM_TRANSPOSE4_PS(mm0, mm1, mm2, mm3);
			_mm_store_ps(p0, mm0);
			_mm_store_ps(p1, mm1);
			_mm_store_ps(p2, mm2);
			_mm_store_ps(p3, mm3);
		}
	}
}

void TestTransposeSSE(int t = 100) {
	int w = 2048;
	int h = 2048;
	ImageT<float> imgSrc, imgDst, imgDstC;
	imgSrc.LoadRawFile<uchar>("../../data/Lena-2048x2048.raw", w, h);
	imgDst.MallocBuffer(imgSrc.height, imgSrc.width);
	imgDstC.MallocBuffer(imgSrc.height, imgSrc.width);
	DWORD tm[4];
	tm[0] = timeGetTime();
	for (int i=0; i<t; i ++)
		TransposeC(imgSrc.buffer, imgSrc.width, imgSrc.height, imgDst.buffer, imgDst.width, imgDst.height);
	tm[1] = timeGetTime();
	for (int i = 0; i < t; i++)
		TransposeSSE(imgSrc.buffer, imgSrc.width, imgSrc.height, imgDstC.buffer, imgDstC.width, imgDstC.height);
	tm[2] = timeGetTime();
	printf("transposeC %d %d, time = %f ms\n", imgDst.width, imgDst.height, float(tm[1] - tm[0])/t);
	printf("TransposeSSE   %d %d, time = %f ms\n", imgDst.width, imgDst.height, float(tm[2] - tm[1]) / t);
	imgSrc.display("src");
	imgDst.display("dst");
	imgDst.display("dstC");
	WaitKey();
}


template<typename T>
	bool BackProjection(const FrameT<T>& _frame, ImageT<T>& img_filted, const Parameter& para, const Matrix2dReal4x4_SSE& mat_proj, Volume& vol)
	{
		bool brtn = true;
		int m(0), n(0);
		const int width = _frame.width;
		const int height = _frame.height;
		const Rect32i* roi = _frame.roi;

		TransposeSSE(_frame.buffer, _frame.width, _frame.height, img_filted.buffer, img_filted.width, img_filted.height);
		//const T fcos = cos(angle_rad);
		//const T fsin = sin(angle_rad);
		//const VolumeCoor<T>   volCor(para.dx, para.dy, para.dz, para.nx, para.ny, para.nz);
		//const DetectorCoor<T> detectorCor(para.du, para.dv, para.nu, para.nv, 
		//	para.offset_nu, 
		//	para.offset_nv);

		//ImageT<T> img_filted(_frame);

		//_frame.display("frame");
		//img_filted.display("img_filted");
		//WaitKey();
		const int MAX_BUFFER_SIZE = 512;
		const int OFFSET_SIZE     = 16;
		float __attribute__((aligned(16))) col[MAX_BUFFER_SIZE + OFFSET_SIZE * 2];
		float __attribute__((aligned(16))) result[MAX_BUFFER_SIZE + OFFSET_SIZE * 2];
		float __attribute__((aligned(16))) result1[MAX_BUFFER_SIZE + OFFSET_SIZE * 2];
		float __attribute__((aligned(16))) xyzw[4 * 512];
		float __attribute__((aligned(16))) xyzwA[4];
		float __attribute__((aligned(16))) xyzwB[4];
		float __attribute__((aligned(16))) xyzwC[4];
		float __attribute__((aligned(16))) xyzwD[4];
		{
			float* p = xyzw;
			for (int i = 0; i < vol.nx; i++, p += 4) *p = i;
		}

		float __attribute__((aligned(16))) ijkw[4] = { 0, 0, 0, 1 };
		for (int k = 0; k < vol.nz; k++) {

			ijkw[2] = k;
					
			//DWORD tm = timeGetTime();
			for(int j = 0 ; j < vol.ny ; j++) {	
				ijkw[1] = j;
				mat_proj.MultipleXYZW<1, 1, 1, 0>(ijkw, xyzw);
				mat_proj.GetXYZW<1, 1, 1, 0>(ijkw, xyzw);

				int nY = int(xyzw[0]);
				float dd  = xyzw[0] - nY;
				float _dd = 1.f - dd;
				int nY1 = nY + 1;
				bool bValid = false;
				if (nY1 < 0) {
				}
				else if (nY1 == 0) {

				}
				else if (nY < img_filted.height) {				
					const float* p0 = &img_filted(0, nY);
					const float* p1 = p0 + img_filted.width;
					float* pDst = col + 16;
					{
						const __m128 mm_DD = _mm_set_ps(_dd, _dd, _dd, _dd);
						const __m128 mmDD =  _mm_set_ps(dd, dd, dd, dd);
						for (int i = 0; i < img_filted.width; i += 4, p0 += 4, p1 += 4, pDst += 4) {
							//__m128 m0 = _mm256_fmadd_ps(_mm256_load_ps(p0), mm_DD, _mm256_mul_ps(_mm256_load_ps(p1), mmDD));
							__m128 m0 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(p0), mm_DD), _mm_mul_ps(_mm_load_ps(p1), mmDD));
							_mm_store_ps(pDst, m0);
						}
					}
					bValid = true;
				}
				else {

				}
				if (1/* bValid*/)
				{
					memcpy(result1 + OFFSET_SIZE, result + OFFSET_SIZE + 1, MAX_BUFFER_SIZE*sizeof(result[0]));
					if (1) {
						if (0) {
							float* pXYZW = xyzw;
							const __m128 m0 = _mm_set_ps(0, k, j, 0); 
							for (int i = 0; i < vol.nx; i++, pXYZW += 4) {
								//__m128 m1 = ;
								_mm_store_ps(pXYZW, _mm_add_ps(_mm_load_ps(pXYZW), m0));
							}
						}

						if (1) {
							__m128 mm2 = mat_proj.mm2;
							__m128 mm3 = mat_proj.mm3;
							float* pXYZW = xyzw;
							for (int i = 0; i < vol.nx; i += 2, pXYZW += 8) {
								__m128 mm5 =  _mm_load_ps(pXYZW);
								__m128 mm6 =  _mm_load_ps(pXYZW + 4);

								__m128 mm7  = _mm_mul_ps(mm2, mm5);
								__m128 mm8  = _mm_mul_ps(mm3, mm5);
								__m128 mm9  = _mm_mul_ps(mm2, mm6);
								__m128 mm10 = _mm_mul_ps(mm3, mm6);

								_MM_TRANSPOSE4_PS(mm7, mm8, mm9, mm10);

								mm5 = _mm_add_ps(_mm_addsub_ps(mm7, mm8), mm9);
								_mm_store_ps(pXYZW, mm5);
							}
						}
					}

					if (0) {
						//__m128 mm1 = mat_proj.mm1;
						__m128 mm2 = mat_proj.mm2;
						__m128 mm3 = mat_proj.mm3;
						__m128 mm5, mm6, mm7;

						float* pXYZW = xyzw;
						for (int i = 0; i < vol.nx; i += 1, pXYZW += 4) {
							mm5 =  _mm_load_ps(pXYZW);
							//ijkw[0] = i;				
							//mm7 = _mm_set_ps(1.f, 1.f, 0, 0);

							//if (VALID_X){
							//	const int MASK_0 = (1<<4) + (1<<5) + (1<<6) + (1<<7) + (1<<0);
							//	mm6 = _mm_dp_ps(mm1, mm5, MASK_0);
							//	mm7 = _mm_or_ps(mm7, mm6);
							//}

							//if (VALID_Y)
							{
								const int MASK_1 = (1 << 4) + (1 << 5) + (1 << 6) + (1 << 7) + (1 << 1);
								mm7 = _mm_dp_ps(mm2, mm5, MASK_1);
								//mm7 = _mm_or_ps(mm7, mm6);
							}

							//if (VALID_Z)
							{
								const int MASK_2 = (1 << 4) + (1 << 5) + (1 << 6) + (1 << 7) + (1 << 0) + (1 << 1) + (1 << 2) + (1 << 3);
								mm6 = _mm_dp_ps(mm3, mm5, MASK_2);
								mm7 = _mm_div_ps(mm7, mm6); 
							}

							_mm_store_ps(xyzwA, mm7);
							//mm5 = _mm_add_ps(mm5, _mm_set_ps(0, 0, 0, 1));
						}
					}

					//for (int i=0; i<vol.nx; i += 4){
					//	mat_proj.GetXYZW<0, 1, 1, 0>(ijkw, xyzwA);
					//	ijkw[0] = i;

					//	mat_proj.GetXYZW<0, 1, 1, 0>(ijkw, xyzwB);
					//	ijkw[0] = i + 1;

					//	mat_proj.GetXYZW<0, 1, 1, 0>(ijkw, xyzwC);
					//	ijkw[0]  = i + 2;

					//	mat_proj.GetXYZW<0, 1, 1, 0>(ijkw, xyzwD);
					//	ijkw[0] = i + 3;

					//}

				}
			}
			//tm = timeGetTime() - tm;
			//std::cout<<"total_time = "<<tm<<std::endl;
		}

		return true;
	
		//	if (para.str_filter == "none"){
		//#if 0
		//		img_filted *= para.cos_image_table.cos_sigma;
		//#else
		//		img_filted /= float(para.nProj)/para.step;
		//#pragma message("do not correct projection image by cos")
		//#endif
		//		//img_filted.display("img_filted filter_none", roi);
		//	}else{
		//		//std::vector<double> filt = filter(para.str_filter, width, 1);
		//#if 1
		//		ImageT<T> img_corrected(_frame);
		//		img_corrected *= para.cos_image_table.cos_sigma;
		//		//img_corrected *= para.cos_image_table.cos_sigma;
		//		//img_corrected *= para.cos_image_table.csc_sigma;
		//#if 1
		//		img_corrected *= para.AngularAndrampFactor;
		//#pragma message("use AngularAndrampFactor")
		//#else
		//#endif
		//		img_corrected.display("img_corrected");
		//		//DWORD tm = timeGetTime();
		//		filterImageRow(para.fft_kernel_real, img_corrected, img_filted);
		//		//tm = timeGetTime() - tm;
		//#else
		//	#pragma message("do not correct projection image by cos")
		//		filterImageRow(para.fft_kernel_real, _frame, img_filted);
		//#endif
		//		img_filted.display("convolutionImageRow", roi);
		//	}

			const T weightB = para.SOD;
		if (roi) img_filted.SetRoi(*roi);
		const int szXY = para.nx*para.ny;
#if 1
#pragma message("use BackProjection SSE")
		//const Matrix2dReal4x4_SSE mat_proj = CalculateProjMatKJI(para, angle_rad);
		{
			//#ifndef _DEBUG
#pragma omp parallel for// private(i,j,k)
			//#endif
					for(int i = 0 ; i < vol.nz ; i++) {
				const int offset_z = szXY*i;
				Matrix2dReal1x4 mat_ijk(0, 0, i, 1), mat_xyz;
				for (int j = 0; j < vol.ny; j++) {
					mat_ijk.data[1] = j;
					const int offset_y = offset_z + vol.nx*j;
					for (int k = 0; k < vol.nx; k++) {
						mat_ijk.data[0] = k;
						const int offset = offset_y + k;
						//MatrixMulti(mat_proj, mat_ijk, mat_xyz);
						mat_proj.Multiple(mat_ijk, mat_xyz);
						const T& disSO_mm = mat_xyz.data[2];
						const T disSO_sc = 1 / disSO_mm;
						mat_xyz.data[0] *= disSO_sc;
						mat_xyz.data[1] *= disSO_sc;

						T& i_det = mat_xyz.data[1];
						T& j_det = mat_xyz.data[0];
						//if (img_filted.IsValid(i_det, j_det))
						{
							//weighted fbp
							vol.buffer[offset] += img_filted(i_det, j_det)*disSO_sc*disSO_sc/**weightB*/;
						}
					}
				}
			}

			return true;
		}
#else
#endif
		return brtn;
	}


template<typename T>
	bool CTBackProjection(ProjectionImageT<T>& proj, const Parameter& para, Volume& img, int proj_index = -1)
	{
		//extern bool cudaCTBackprojectionReal32(ProjectionImage& proj, const Parameter& para, Volume& img);
		//return cudaCTBackprojectionReal32(proj, para, img);
		bool bRtn = true;
		if (proj_index == -1) {
			DestroyAllWindow();
			img.Zeros();
		}
		Volume vol(img.nx, img.ny, img.nz);
		DWORD total_time = 0;

		int nStart = 0;
		Rect32i rcSubImg(96, 96, para.nu - 96 * 2, para.nv - 96 * 2);
		int step = para.step;
		int nProjCount = para.nProj;
		if (proj_index >= 0) {
			nStart = proj_index;
			nProjCount = 1;
		}
		ImageT<T> img_filted;
		for (int idx = 0; idx < nProjCount; idx += step) {
			int i = idx + nStart;
			if (i >= para.nProj) i -= para.nProj;
			const int& frame_index = i;
			FrameT<T> frame;
			bRtn = proj.GetFrame(frame, frame_index);
			VERIFY_TRUE(bRtn);
			if (!bRtn) break;	
			//frame.display("sub_frame_96-96", &rcSubImg);
			float angle_rad = para.direction*para.vec_deg[frame_index] / 180.0*PI + para.volume_postion_angle;
			const Matrix2dReal4x4_SSE mat_proj = CalculateProjMatKJI(para, angle_rad);
			frame.display("frame");
			img_filted.MallocBuffer(frame.height, frame.width);
			DWORD tm = timeGetTime();
			bRtn = BackProjection<T>(frame, img_filted, para, mat_proj, img);
			VERIFY_TRUE(bRtn);
			if (!bRtn) break;
			tm = timeGetTime() - tm;
			total_time += tm;
			if (i != 0) ClearCurrentLine();
			printf("CTBackProjection(%03d/%03d=%.1f%s, time=%d ms/%.3f m, roi(x(%d) y(%d))",
				i,
				para.nProj,
				float(i + 1)/para.nProj*100,
				"%",
				tm,
				float(total_time)/(60 * 1000),
				proj.roi?proj.roi->x:0,
				proj.roi?proj.roi->y:0);
			//img += vol;

			vol = img;

#if 0
			vol.SliceX(img.nx / 8);
			vol.SliceY(img.ny / 8);
			vol.SliceZ(img.nz / 2);
#endif
		}
		printf("\n");

		return bRtn;
	}

bool CreateCoordinateTable(const Parameter& para) {
	return false;
}



void Reconstruction() {
	DISPLAY_FUNCTION;
	char szPath[256]; 
	//////////////////////////////////////////////////
	Parameter para;
	para.Display();
	ProjectionImageT<float> proj;
	Volume volum(para.nx, para.ny, para.nz);
	proj.LoadFolder<float>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj, para.step);

	//Rect32i roi(1, 1, para.nu-1*2, para.nv-1*2);
	//proj.SetRoi(roi);
	CTBackProjection<float>(proj, para, volum);

	sprintf(szPath, "%s%s/proj_%d.volume", para.str_working_folder.c_str(), "volume", para.nProj);
	//volum.Save(szPath);
}



int main0()
{
	/*   __m256 a, b, c, d;
	   int i;
	   for (i = 0; i < 8; i++) {
	       a.m256_f32[i] = i;
	       b.m256_f32[i] = 2.;
	       c.m256_f32[i] = 3.;
	   }
	   d = _mm256_nmsub_ps(a, b, c);
	   for (i = 0; i < 8; i++) printf_s(" %.3f", d.m256_f32[i]);
	   printf_s("\n");*/
	return 0;
}

int main(int argc, char* argv[])
{
	//TestMatrixMulti();
	TestTransposeSSE();
	//Reconstruction();
	return 0;
}