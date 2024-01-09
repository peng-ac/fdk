#include <iostream>
#include "mpiFDK.h"

Parameter& Get608x616x512(
	std::string sSrcDir = "../../data/proj-img/", 
	std::string sDstDir = "../../working/flt-proj-img/", 
	int nx = 512,
	int ny = 512,
	int nz = 512)
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

Parameter& Get2048x2048x1200(int nx = 1024, int ny = 1024, int nz = 1024)
{
	static std::shared_ptr<Parameter> pPara;
	pPara = std::make_shared<Parameter>();
	Parameter& para = *pPara;
	para.du = para.nu*para.du / 2048; para.dv = para.nv*para.dv / 2048;
	para.nu = para.nv = 2048; para.nProj = 1200;
	
	double fx = double(nx) / para.nx; 
	double fy = double(ny) / para.ny;
	double fz = double(nz) / para.nz;
	para.nx *= fx; para.ny *= fy; para.dx /= fx; para.dy /= fy; para.nz *= fz; para.dz /= fz;
			
	para.init();
	//para.str_proj_img_folder = "../../data/phantom3d/filtered-shepp-logan_w2048_h2048_c1200/";
	para.str_proj_img_folder = "../../data/phantom3d/shepp-logan_w2048_h2048_c1200/";
	para.str_working_folder = "../../working/filted-shepp-logan_w2048_h2048_c1200/";
	//para.Display();
	return para;
}

int cuFDK(int argc, char **argv)
{
	Parameter& para = Get608x616x512();
	//Parameter& para = Get2048x2048x1200();
	StopWatchWin watch;
	watch.start();
	{
		MPI_Init(&argc, &argv);
		//MpiWraper mpi(para, group);
		//mpi.Run();
		{
			mpiFDK fdk(para);	
			fdk.Run();
		}	
		MPI_Finalize();
	}
	watch.stop();
	std::cout << "total time : " << watch.getAverageTime()/1000 <<" s" <<std::endl;
	return 0;
}


