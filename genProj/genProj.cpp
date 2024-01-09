#include <iostream>
#include <stdio.h>
#include "cudaIterType.cuh"
#include "../common/cudaLib.cuh"
#include "../common/type.h"

using namespace std;

Parameter& GetParameter(int nu = 608, int nv = 616, int nProj = 512)
{
	static std::shared_ptr<Parameter> pPara;
	pPara = std::make_shared<Parameter>();
	Parameter& para = *pPara;
	para.du = para.nu*para.du / nu; para.dv = para.nv*para.dv / nv;
	para.nu = nu; para.nv = nv; para.nProj = nProj;
	
	//double fx = double(nx) / para.nx; 
	//double fy = double(ny) / para.ny;
	//double fz = double(nz) / para.nz;
	//para.nx *= fx; para.ny *= fy; para.dx /= fx; para.dy /= fy; para.nz *= fz; para.dz /= fz;
	//para.nx *= 2; para.ny *= 2; para.dx /= 2; para.dy /= 2; para.nz *= 4; para.dz /= 4;
	//para.nx *= 2; para.ny *= 2; para.dx /= 2; para.dy /= 2; para.nz *= 2; para.dz /= 2;
			
	para.init();
	para.str_proj_img_folder = "/tmp/";
	para.str_working_folder = "/tmp/";
	//para.Display();
	return para;
}

static void ForwardProjection(const char* szDstDir, const char* szVolume, int nu = 608, int nv = 616, int nProj = 512)
{
	DISPLAY_FUNCTION;
	printf("szDstDir = %s\n", szDstDir);
	printf("szVolume = %s\n", szVolume);
	printf("nu=%d, nv=%d, nProj=%d\n", nu, nv, nProj);
	
	bool bRtn;
	Parameter& para = GetParameter(nu, nv, nProj);
	para.Display();
	
	char pWorkFolder[1024];
	printf("szDstDir=%s\n", szDstDir);
	VERIFY_TRUE(IsPathExisted(szDstDir));
	VERIFY_TRUE(IsPathExisted(szVolume));

	sprintf(pWorkFolder, "%sshepp-logan_w%d_h%d_c%04d/",
		szDstDir,
		para.nu,
		para.nv,
		para.nProj);
	std::cout << "pWorkFolder=" << pWorkFolder << std::endl;
	ProjectionImage proj_ori, proj_fp;
	Volume volume(para.nx, para.ny, para.nz);

	//VERIFY_TRUE(proj_ori.LoadFolder<float>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj, para.step, 0));
	//VERIFY_TRUE(proj_fp.MallocBuffer(para.nu, para.nv, para.nProj, para.step).IsValid());
	VERIFY_TRUE(volume.Load(szVolume));
	volume.SetUint(para.dx, para.dy, para.dz);
	volume.SliceZ(volume.nz / 2);
	//WaitKey();
	std::vector<DevData<float>> vec_dev_proj(para.nProj), vec_dev_forward_proj(para.nProj);
	DevArray3D<float> dev_vol(volume.nx, volume.ny, volume.nz/*, cudaArraySurfaceLoadStore*/);
	dev_vol.CopyFromHost(volume.buffer, volume.nx, volume.nx, volume.ny, volume.nz);
	
	ImageT<float> image(para.nu, para.nv);
	DevData<float> dev_image(para.nu, para.nv);

	cudaDeviceSynchronize();

	for (int i = 0; i < para.nProj; i++) {
		cudaCTForwardProjection_f32(proj_ori, para, volume, &dev_vol, NULL, image, dev_image, &vec_dev_forward_proj, NULL, i); 
		char szPath[1024];
		sprintf(szPath, "%simg%04d.raw", pWorkFolder, i);
		//VERIFY_TRUE(image.Save(szPath));
		WriteToFile<float, float>(szPath, image.buffer, image.width*image.height, true);
		usleep(50);
	}


}

int main(int argc, char **argv)
{
	DISPLAY_FUNCTION;
	for (int i = 0; i < argc; i++){
		std::cout << argv[i] << " ";
	}
	std::cout << std::endl;
	
	VERIFY_TRUE(argc == 6);
#if 0
	char szDstDir[1024] = "";
	char szVolume[1024] = "/home/chen/projects/data/phantom3d/Shepp-Logan-512x512x512.vol";
	int nu = 608;
	int nv = 616;
	int nProj = 512
#else
	char* szDstDir = argv[1];
	char* szVolume = argv[2];
	int nu = atoi(argv[3]);
	int nv = atoi(argv[4]);
	int nProj = atoi(argv[5]);
#endif
	ForwardProjection(szDstDir, szVolume, nu, nv, nProj);
	return 0;
}
