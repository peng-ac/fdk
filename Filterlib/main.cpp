#include "FilterLib.h"
#include "../common/StopWatch.h"

int test_single_thread(const Parameter& para) {
	DISPLAY_FUNCTION;
	
	VERIFY_TRUE(IsPathExisted(para.str_proj_img_folder.c_str()));
	const char* pSrc = para.str_proj_img_folder.c_str();
	const char* pDst = para.str_working_folder.c_str();
	
	int order = log(para.fft_kernel_real.size()) / log(2) + 0.5;
	ProjectionImageT<float> proj;
	std::cout << "begin proj.LoadFolder" << std::endl;
	proj.LoadFolder<float>(pSrc, para.nu, para.nv, para.nProj, para.step);
	std::cout << "end proj.LoadFolder" << std::endl;
	//proj.Display("proj");

	ProjectionImageT<float> proj_filter;
	std::cout << "begin proj_filter.MallocBuffer" << std::endl;
	proj_filter.MallocBuffer(para.fft_kernel_real.size(), proj.height, proj.GetCount(), 1);
	std::cout << "begin proj_filter.MallocBuffer" << std::endl;
	proj_filter.Zero();

	std::cout << "begin ippsFFT fft(order)" << std::endl;
	ippsFFT fft(order);
	std::cout << "begin ippsFFT fft(order)" << std::endl;
	
	ImageT<float> img_imag;
	img_imag.MallocBuffer(proj_filter.width, proj_filter.height);
	Rect32i rect((proj_filter.width - proj.width) / 2, 0, proj.width, proj.height);
	proj_filter.SetRoi(rect);
	double total = 0;
	for (int i = 0; i < proj_filter.GetCount(); i++) {
		FrameT<float> src = proj.GetFrame(i);
		FrameT<float> dst = proj_filter.GetFrame(i);
		src.display("dst_ori");
		FrameT<float> dst_imag = img_imag.GetFrame();
		DWORD tm = timeGetTime();
		StopWatchWin watch;
		watch.start();
		FilterCosCorrect(src, para.cos_image_table.cos_sigma.GetFrame(), src);
		FilterImageRow(src, dst, dst_imag, para.fft_kernel_real_f32, fft);
		watch.stop();
		total += watch.getAverageTime();
		//tm = timeGetTime() - tm;
		std::cout << "time = " << watch.getAverageTime() << std::endl;
		dst.display("dst_filter");
		//WaitKey();
	}
	std::cout << "avg-time = " << total / proj_filter.GetCount() << std::endl;
	proj_filter.SaveToFolder(pDst);

	WaitKey();


	return 0;
}

template<int threads>
	int test_multi_thread(const Parameter& para) {
		DISPLAY_FUNCTION;
	
		const char* pSrc = para.str_proj_img_folder.c_str();
		const char* pDst = para.str_working_folder.c_str();
		
		int order = log(para.fft_kernel_real.size()) / log(2) + 0.5;
		ProjectionImageT<float> proj;
		proj.LoadFolder<float>(pSrc, para.nu, para.nv, para.nProj, para.step);

		//proj.Display("proj");

		ProjectionImageT<float> proj_filter;
		proj_filter.MallocBuffer(para.fft_kernel_real.size(), proj.height, proj.GetCount(), 1);
		proj_filter.Zero();
	
		FilterImageRowMulti<threads> FilterImageRow(order);
		ippsFFT fft(order);
		ImageT<float> img_imag;
		img_imag.MallocBuffer(proj_filter.width, proj_filter.height);
		Rect32i rect((proj_filter.width - proj.width) / 2, 0, proj.width, proj.height);
		proj_filter.SetRoi(rect);
		double total = 0;
		for (int i = 0; i < proj_filter.GetCount(); i++) {
			FrameT<float> src = proj.GetFrame(i);
			FrameT<float> dst = proj_filter.GetFrame(i);
			src.display("dst_ori");
			FrameT<float> dst_imag = img_imag.GetFrame();
			DWORD tm = timeGetTime();
			StopWatchWin watch;
			watch.start();
			FilterCosCorrect(src, para.cos_image_table.cos_sigma.GetFrame(), src);
			FilterImageRow(src, dst, dst_imag, para.fft_kernel_real_f32);
			watch.stop();
			total += watch.getAverageTime();
			//tm = timeGetTime() - tm;
			std::cout << "time = " << watch.getAverageTime() << std::endl;
			dst.display("dst_filter");
			//WaitKey();
		}
		std::cout << "avg-time = " << total / proj_filter.GetCount() << std::endl;
		//proj_filter.SaveToFolder(pDst);

		WaitKey();


		return 0;
	}

template<int threads>
	int test_multi_data(const Parameter& para) {
	DISPLAY_FUNCTION;
	
	const char* pSrc = para.str_proj_img_folder.c_str();
	const char* pDst = para.str_working_folder.c_str();
	
	int order = log(para.fft_kernel_real.size()) / log(2) + 0.5;
	ProjectionImageT<float> proj;
	proj.LoadFolder<float>(pSrc, para.nu, para.nv, para.nProj, para.step);

	//proj.Display("proj");

	ProjectionImageT<float> proj_filter;
	proj_filter.MallocBuffer(para.fft_kernel_real.size(), proj.height, proj.GetCount(), 1);
	proj_filter.Zero();

	std::vector<std::shared_ptr<FilterImageRowMulti<1>>> vecFilterImageRow(threads);  //(order);
	std::vector<ImageT<float>> vec_img_imag(threads);
	for (int i = 0; i < threads; i++){
		vecFilterImageRow[i] = std::make_shared<FilterImageRowMulti<1>>(order);
		vec_img_imag[i].MallocBuffer(proj_filter.width, proj_filter.height);		
	}

	Rect32i rect((proj_filter.width - proj.width) / 2, 0, proj.width, proj.height);
	proj_filter.SetRoi(rect);
	double total = 0;
	
	StopWatchWin watch;
	watch.start();
	#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < proj_filter.GetCount(); i++) {
		int tid = omp_get_thread_num();
		ImageT<float>& img_imag = vec_img_imag[tid];
		FilterImageRowMulti<1>& FilterImageRow = *vecFilterImageRow[tid];
		FrameT<float> src = proj.GetFrame(i);
		FrameT<float> dst = proj_filter.GetFrame(i);
		src.display("dst_ori");
		FrameT<float> dst_imag = img_imag.GetFrame();
		FilterCosCorrect(src, para.cos_image_table.cos_sigma.GetFrame(), src);
		FilterImageRow(src, dst, dst_imag, para.fft_kernel_real_f32);
		ippsCopy(dst.buffer + rect.x, dst.width, src.buffer, src.width, src.width, src.height);
		//WaitKey();
	}	
	watch.stop();
	total += watch.getAverageTime();
	//tm = timeGetTime() - tm;
	std::cout << "total : "<<watch.getAverageTime() << " ms/" << proj_filter.GetCount() << "frames"<<std::endl;
	std::cout << "time = " << watch.getAverageTime() / proj_filter.GetCount() << "ms/frame"<<std::endl;
	
	//dst.display("dst_filter");
	//proj_filter.SaveToFolder(pDst);
	proj.SaveToFolder(pDst);

	WaitKey();


	return 0;
}

int test_filter_engine(const Parameter& para) {
	DISPLAY_FUNCTION;
	const char* pSrc = para.str_proj_img_folder.c_str();
	const char* pDst = para.str_working_folder.c_str();
	
	int order = log(para.fft_kernel_real.size()) / log(2) + 0.5;
	ProjectionImageT<float> proj;
	proj.LoadFolder<float>(pSrc, para.nu, para.nv, para.nProj, para.step);

	//proj.Display("proj");
	
	ProjectionImageT<float> proj_filter;
	proj_filter.MallocBuffer(proj.height, proj.width, proj.GetCount(), 1);
	proj_filter.Zero();

	double total = 0;
	auto frm = para.cos_image_table.cos_sigma.GetFrame();
	FilterEngine flt(frm, para.fft_kernel_real_f32);
	for (int i = 0; i < proj_filter.GetCount(); i++) {
		FrameT<float> src = proj.GetFrame(i);
		FrameT<float> dst = proj_filter.GetFrame(i);		
		StopWatchWin watch;
		watch.start();
		flt.Filtering(src, dst, true);
		watch.stop();
		total += watch.getAverageTime();

		//ClearCurrentLine();
		std::cout << "time = " << watch.getAverageTime()<<std::endl;
		dst.display("dst_filter");
	}
	std::cout << "avg-time = " << total / proj_filter.GetCount() << std::endl;
	proj_filter.SaveToFolder(pDst);

	WaitKey();


	return 0;
}



//int main(int argc, char** argv)
//{
//	//return test_filter_engine(para);
//	
//	printf("Filterlib src-path dst-path thread-count\n");
//	if (argc < 4) {
//		std::cout << "error in arguments" << std::endl;
//		return 1;
//	}
//		
//	const char* src = argv[1];
//	const char* dst = argv[2];
//	const char* pthreads = argv[3];
//	int nthread = atoi(pthreads);
//	if (nthread == 1) test_single_thread(para);
//	else {
//		if (0){
//			if (nthread == 2) test_multi_thread<2>(para);
//			else if (nthread == 4) test_multi_thread<4>(para);
//			else if (nthread == 8) test_multi_thread<8>(para);
//			else if (nthread == 16) test_multi_thread<16>(para);
//			else if (nthread == 32) test_multi_thread<32>(para);
//			else if (nthread == 36) test_multi_thread<36>(para);
//			else if (nthread == 64) test_multi_thread<64>(para);
//			else {	
//			}
//		}
//		if (1){
//			if (nthread == 2) test_multi_data<2>(para);
//			else if (nthread == 4) test_multi_data<4>(para);
//			else if (nthread == 8) test_multi_data<8>(para);
//			else if (nthread == 16) test_multi_data<16>(para);
//			else if (nthread == 32) test_multi_data<32>(para);
//			else if (nthread == 36) test_multi_data<36>(para);
//			else if (nthread == 64) test_multi_data<64>(para);
//			else if (nthread == 128) test_multi_data<128>(para);
//			else if (nthread == 120) test_multi_data<8>(para);
//			else if (nthread == 80) test_multi_data<80>(para);
//			else {	
//			}				
//		}
//	}
//	//test_multi_thread(para);
//	//test_single_thread(para);
//	//test_multi_thread(para);
//	return 0;
//}

Parameter& GetDigitalPhantom(int nx = 512,
	int ny = 512,
	int nz = 512, 	
	int nu = 2048,
	int nv = 2048,
	int nProj = 512,
	std::string sSrcDir = "../../data/proj-img/", 
	std::string sDstDir = "../../working/flt-proj-img/",
	int TOTAL_PROJ = 16384,
	int step = 0)
{
	static std::shared_ptr<Parameter> pPara;
	pPara = std::make_shared<Parameter>();
	Parameter& para = *pPara;
	para.du = para.nu*para.du / nu; para.dv = para.nv*para.dv / nv;
	para.nu = nu; para.nv = nv; para.nProj = nProj;
	
	VERIFY_TRUE(int(std::pow(2, std::log2(nProj)) + 0.5) == nProj);
	para.step = step == 0?TOTAL_PROJ / nProj:step;
	//para.nProj /= 2; para.step *= 2;
	
	double fx = double(nx) / para.nx; 
	double fy = double(ny) / para.ny;
	double fz = double(nz) / para.nz;
	para.nx *= fx; para.ny *= fy; para.dx /= fx; para.dy /= fy; para.nz *= fz; para.dz /= fz;
			
	para.init();
	//para.str_proj_img_folder = "../../data/phantom3d/filtered-shepp-logan_w2048_h2048_c1200/";
	para.str_proj_img_folder = sSrcDir + StringFormat("shepp-logan_w%d_h%d_c%d/", nu, nv, TOTAL_PROJ);
	VERIFY_TRUE(IsPathExisted(para.str_proj_img_folder.c_str()));
	para.str_working_folder = sDstDir;
	//para.Display();
	return para;
}

int main(int argc, char** argv)
{
	std::string str;
	printf("argc=%d : ", argc);
	for (int i = 0; i < argc; i++){
		str += StringFormat("%s ", argv[i]);
	}
	printf("Main_CMD : %s\n", str.c_str());
	VERIFY_TRUE(argc == 7);
	if (argc != 7) return 1;
	
	int nu, nv;
	int projs;
	std::string src_dir, dst_dir, threads("1");
	nu = atoi(argv[1]);
	nv = atoi(argv[2]);
	projs = atoi(argv[3]);
	src_dir = argv[4];
	dst_dir = argv[5];	
	threads = argv[6];	 
		
	Parameter& para = GetDigitalPhantom(512, 512, 512, nu, nv, projs, src_dir, dst_dir, 16384, 1);
	para.Display();
	
	
	int nthread = atoi(threads.c_str());
	if (nthread == 1) test_single_thread(para);
	else {
			if (0) {
				if (nthread == 2) test_multi_thread<2>(para);
				else if(nthread == 4) test_multi_thread<4>(para);
				else if(nthread == 8) test_multi_thread<8>(para);
				else if(nthread == 16) test_multi_thread<16>(para);
				else if(nthread == 32) test_multi_thread<32>(para);
				else if(nthread == 36) test_multi_thread<36>(para);
				else if(nthread == 64) test_multi_thread<64>(para);
				else {	
					}
			}
			if (1) {
				if (nthread == 2) test_multi_data<2>(para);
				else if(nthread == 4) test_multi_data<4>(para);
				else if(nthread == 8) test_multi_data<8>(para);
				else if(nthread == 16) test_multi_data<16>(para);
				else if(nthread == 32) test_multi_data<32>(para);
				else if(nthread == 36) test_multi_data<36>(para);
				else if(nthread == 64) test_multi_data<64>(para);
				else if(nthread == 128) test_multi_data<128>(para);
				else if(nthread == 120) test_multi_data<8>(para);
				else if(nthread == 80) test_multi_data<80>(para);
				else {	
					}				
			}
		}
	//test_multi_thread(para);
	//test_single_thread(para);
	//test_multi_thread(para);
	return 0;
}