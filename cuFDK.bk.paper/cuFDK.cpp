#include <iostream>
#include "mpiFDK.h"
#include "../common/cudaLib.cuh"

Parameter& Get608x616x512(
	int nx = 512*2,
	int ny = 512*2,
	int nz = 512*2, 	
	int nProj = 512,
	std::string sSrcDir = "../../data/proj-img/", 
	std::string sDstDir = "../../working/flt-proj-img/")
{
	static std::shared_ptr<Parameter> pPara;
	pPara = std::make_shared<Parameter>();
	Parameter& para = *pPara;
	para.nu = 608; para.nv = 616; para.nProj = 512 / 1;
	double fx = double(nx) / para.nx; 
	double fy = double(ny) / para.ny;
	double fz = double(nz) / para.nz;
	para.nx *= fx; para.ny *= fy; para.dx /= fx; para.dy /= fy; para.nz *= fz; para.dz /= fz;

	para.init();
	para.str_proj_img_folder = sSrcDir;
	para.str_working_folder = sDstDir;
	//para.Display();
	return para;
}

Parameter& Get2048x2048x1200(int nx = 512,
	int ny = 512,
	int nz = 512, 	
	int nu = 2048,
	int nv = 2048,
	int nProj = 512,
	std::string sSrcDir = "../../data/proj-img/", 
	std::string sDstDir = "../../working/flt-proj-img/",
	int TOTAL_PROJ = 1200)
{
	static std::shared_ptr<Parameter> pPara;
	pPara = std::make_shared<Parameter>();
	Parameter& para = *pPara;
	para.du = para.nu*para.du / 2048; para.dv = para.nv*para.dv / 2048;
	para.nu = para.nv = 2048; para.nProj = 1200;
	
	VERIFY_TRUE(int(std::pow(2, std::log2(nProj)) + 0.5) == nProj);
	para.step = TOTAL_PROJ / nProj;
	//para.nProj /= 2; para.step *= 2;
	
	double fx = double(nx) / para.nx; 
	double fy = double(ny) / para.ny;
	double fz = double(nz) / para.nz;
	para.nx *= fx; para.ny *= fy; para.dx /= fx; para.dy /= fy; para.nz *= fz; para.dz /= fz;
			
	para.init();
	para.str_proj_img_folder = "../../data/phantom3d/filtered-shepp-logan_w2048_h2048_c1200/";
	//para.str_proj_img_folder = sSrcDir;
	para.str_working_folder = sDstDir;
	//para.Display();
	return para;
}

Parameter& GetDigitalPhantom(int nx = 512,
	int ny = 512,
	int nz = 512, 	
	int nu = 2048,
	int nv = 2048,
	int nProj = 512,
	std::string sSrcDir = "../../data/proj-img/", 
	std::string sDstDir = "../../working/flt-proj-img/",
	int TOTAL_PROJ = 16384)
{
	static std::shared_ptr<Parameter> pPara;
	pPara = std::make_shared<Parameter>();
	Parameter& para = *pPara;
	para.du = para.nu*para.du / nu; para.dv = para.nv*para.dv / nv;
	para.nu = nu; para.nv = nv; para.nProj = nProj;
	
	VERIFY_TRUE(int(std::pow(2, std::log2(nProj)) + 0.5) == nProj);
	para.step = TOTAL_PROJ / nProj;
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

inline void RemoveMpiOverhead(Mpi2D& mpi2d)
{
	size_t buf_size = size_t(1024) * 1024 * 1024 * 2;
	BufferT<float> src, dst;
	size_t repeat = 1;
	double t1;
	for (int s = 0; s < repeat; s++) {
		double tt1 = MPI_Wtime();
		for (size_t t = 0; t < buf_size; t += INT_MAX) {
			auto pSnd = src.data() + t;
			auto pRcv = dst.data() + t;
			size_t count = MIN(INT_MAX, buf_size - t);
			MPI_Reduce(pSnd, pRcv, count, MPI_FLOAT, MPI_SUM, 0, mpi2d.col.comm);	
		}	
		t1 += MPI_Wtime() - tt1;		
	}
	t1 /= repeat;	
}

inline void RemoveOverhead(int argc, char **argv)
{
	StopWatchWin w1;
	w1.start();
	int count;
	cudaGetDeviceCount(&count);
	for (int i = count - 1; i >= 0; i--){
		cudaSetDevice(i);
		GpuBuffer<float> data(1);
	}
	w1.stop();

	
	printf("cudaSetDevice time=%f\n", w1.getAverageTime());
}

int cuFDK(int argc, char **argv)
{	
	LogInfo logInfo;
	int ranks_per_node = 2;
	int rows = ranks_per_node;
	int volume_size = 512;	
	int nx(volume_size), ny(volume_size), nz(volume_size);
	int nu, nv;
	int projs;
	std::string src_dir, dst_dir;
	if (argc > 1) ranks_per_node = atoi(argv[1]);
	if (argc > 2) rows = atoi(argv[2]);
	if (argc > 3) nx = atoi(argv[3]);
	if (argc > 4) ny = atoi(argv[4]);
	if (argc > 5) nz = atoi(argv[5]);	
	if (argc > 6) nu = atoi(argv[6]);
	if (argc > 7) nv = atoi(argv[7]);
	if (argc > 8) projs = atoi(argv[8]);
	if (argc > 9) src_dir = argv[9];
	if (argc > 10) dst_dir = argv[10];
	RemoveOverhead(argc, argv);
	float total = 0;
	StopWatchWin watch;
	StopWatchWin w1;
	//Parameter& para = Get608x616x512("../../data/proj-img/", "../../working/flt-proj-img/", 512*scale, 512*scale, 512*scale);
	//Parameter& para = Get2048x2048x1200(1024, 1024, 1024);
	//Parameter& para = Get2048x2048x1200(volume_size, volume_size, volume_size);
	Parameter& para = GetDigitalPhantom(nx, ny, nz, nu, nv, projs, src_dir.c_str(), dst_dir.c_str(), 16384);
	//Parameter& para = Get2048x2048x1200(volume_size, volume_size, volume_size, 2048, 2048, 1200, src_dir.c_str(), dst_dir.c_str(), 1200);	
	{	
		watch.start();
		{			
			w1.start();
			MPI_Init(&argc, &argv); 
			MPI_Barrier(MPI_COMM_WORLD);
			w1.stop();
			total += w1.getAverageTime();
			logInfo.fMpiInitTime = w1.getAverageTime();
			printf("MPI_Init time =%f ms, [(%d,%d)x%d]->[%d,%d,%d], step=%d\n", w1.getAverageTime(), para.nu, para.nv, para.nProj, para.nx, para.ny, para.nz, para.step);
			if (1){
				int world_size;
				MPI_Comm_size(MPI_COMM_WORLD, &world_size);
				if (world_size == 0)
					para.Display();				
			}
			{
				mpiFDK fdk(para, logInfo, ranks_per_node, rows);	
				fdk.Run();
				{
					float* pSnd = NULL;
					float* pRcv = NULL;
					int count = 0;				
					if (fdk.GetXY().y == 0){
						float fStoreTime = 0;
						pSnd = &logInfo.fStoreTime;
						pRcv = &fStoreTime;
						MPI_Reduce(pSnd, pRcv, 1, MPI_REAL, MPI_SUM, 0, fdk.mpi2d.row.comm);	 
						logInfo.fStoreTime = fStoreTime / fdk.mpi2d.row.size;
					}
					{
						const int COUNT = 10;
						float vec[COUNT] = { 0, };
						pSnd = &logInfo.fFilterInitTime;
						pRcv = vec;
						MPI_Reduce(pSnd, pRcv, COUNT, MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
						for (int i = 0; i < COUNT; i++){
							vec[i] /= fdk.mpi2d.size;
						}
						if (fdk.mpi2d.rank == 0){
							memcpy(pSnd, vec, sizeof(vec));
						}
					}
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			w1.start();
			MPI_Finalize();
			w1.stop();
			logInfo.fFinalizeTime = w1.getAverageTime();
			printf("rank=%d,MPI_Finalize time =%f ms\n", logInfo.rank, w1.getAverageTime());
		}
		watch.stop();
	}
	logInfo.fTotalTime = watch.getAverageTime();
	std::cout << "total time : " << watch.getAverageTime()/1000 <<" s" <<std::endl;
	std::cout<<"--------------------------------------------------"<<std::endl;
	if (logInfo.rank == 0){
		logInfo.Show(para);
	}
	return 0;
}


