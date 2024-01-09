#include <iostream>
#include "IppLib.h"
#include "../common/type.h"
#include "StopWatch.h"


using namespace std;

#if 0
int main(int argc, char *argv[])
{
	char sz[] = "Hello, World!"; 	//Hover mouse over "sz" while debugging to see its contents
	cout << sz << endl; 	//<================= Put a breakpoint here
	return 0;
}
#else


int main(int argc, char* argv[])
{
	int W = 512;
	int H = 512;
	ImageT<float> img_r, img_i, img_r_ori, img_i_ori;
	img_r.MallocBuffer(W, H);
	img_r_ori.MallocBuffer(W, H);
	assert(img_r.LoadRawFile<uchar>("../../data/Lena.512x512.raw", W, H) == true);
	//img_r.LoadImageFile("c:/lena.jpg");
	//img_r_ori.LoadImageFile("c:/lena.jpg");

	//img_i.MallocBuffer(img_r.width, img_r.height).Clear();
	//img_i_ori.MallocBuffer(img_r.width, img_r.height).Clear();
	//img_r.display("img_r");
	std::vector<ImageT<float>> vecImg_r(512);
	std::vector<ImageT<float>> vecImg_i(512);

#pragma omp parallel for
	for (int i = 0; i < vecImg_r.size(); i++) {
		//vecImg_r[i].LoadImageFile("c:/lena.jpg");
		vecImg_i[i].MallocBuffer(512, 512);
		vecImg_i[i].MallocBuffer(vecImg_r[i].width, vecImg_r[i].height).Clear();
	}

	DWORD tm = timeGetTime();
	//#pragma omp parallel for
	for(int s = 0 ; s < vecImg_r.size() ; s++) {
		ImageT<float>& img_r = vecImg_r[s];
		ImageT<float>& img_i = vecImg_i[s];

		ippsFFT fft(10, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast);	
		//img_i.Clear();
		//#pragma omp parallel for
		for(int i = 0 ; i < img_r.height ; i++) {
			fft.FFTFwd_CToC_32f_I(img_r.buffer + i*img_r.width, img_i.buffer + i*img_r.width); 
			fft.FFTInv_CToC_32f_I(img_r.buffer + i*img_r.width, img_i.buffer + i*img_r.width); 
			//FFT(1, fft.order, img_r_ori.buffer + i*img_r.width, img_i_ori.buffer + i*img_r.width);
			//FFT(-1, fft.order, img_r_ori.buffer + i*img_r.width, img_i_ori.buffer + i*img_r.width);
		}
	}
	tm = timeGetTime() - tm;
	std::cout << "tm = " << tm << std::endl;
	for (int s = 0; s < MIN(1, vecImg_r.size()); s++) {
		ImageT<float>& img_r = vecImg_r[s];
		img_r.display("fft_img_r");
		//	fft.FFTInv_CToC_32f_I(img_r.buffer + i*img_r.width, img_i.buffer + i*img_i.width); 
	}
	

	//ImageT<float> dif = img_r;
	//Sub_Vec(img_r.buffer, img_r_ori.buffer, dif.buffer, dif.width*dif.height);
	//float fMax, fMin;
	//GetMaxMin(dif.buffer, dif.width*dif.height, fMax, fMin);

	WaitKey();

	return 0;
}

int mainT(int argc, char* argv[]) {
	Ipp8u src[8 * 4] = { 1, 2, 3, 4, 8, 8, 8, 8, 1, 2, 3, 4, 8, 8, 8, 8, 1, 2, 3, 4, 8, 8, 8, 8, 1, 2, 3, 4, 8, 8, 8, 8 }; 
	Ipp8u dst[4 * 4]; 
	IppiSize srcRoi = { 4, 4 }; 
	ippiTranspose_8u_C1R(src, 8, dst, 4, srcRoi);
	return 0;
}

#if 0
inline void IppRotate(const float* src, float* dst, int width, int height, double xCenter, double yCenter, double angle)
{
	double xShift, yShift;
	ippiGetRotateShift(xCenter, yCenter, angle, &xShift, &yShift);
	IppiSize sz = { width, height };
	IppiRect roi = { 0, 0, width, height };
	int nStep = sizeof(src[0])*width;
	//ippiRotate_32f_C1R(src, sz, nStep, roi, dst, nStep, roi, angle, xShift, yShift, IPPI_INTER_LINEAR); 
	ippiRotate_32f_C1R(src, sz, nStep, roi, dst, nStep, roi, angle, xShift, yShift, IPPI_INTER_CUBIC); 
}


int main(int argc, char* argv[]) {
	ImageT<float> img, imgDst;
	img.LoadRawFile("c:/tmp/IMG0000.raw", 1280, 1024);
	//img.LoadRawFile("C:/temp/proj-img/IMG0000.raw", 608, 616);
	imgDst.MallocBuffer(img.width, img.height);
	//StopWatchWin watch;
	//watch.start();
	for(int i = 0 ; i < 100 ; i++)
		IppRotate(img.buffer, imgDst.buffer, img.width, img.height, img.width / 2.0, img.height / 2.0, 45);
	//watch.stop();
	//std::cout << "rot time = " << watch.getAverageTime() << " ms" << std::endl;
	imgDst.display("imgDst");
	img.display();
	WaitKey();

	return 0;
}
#endif

#endif
