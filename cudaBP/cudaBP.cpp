#include <iostream>
#include "../common/type.h"
#include "../common/StopWatch.h"
#include "backprojection.cuh"

template<typename T>
T* NoConstant(const T* ptr){
	T* p = 0;
	memcpy(&p, &ptr, sizeof(T*));
	return p;
}

using namespace std;

template<typename BackprojectionType>
	static void Test(int TRANPOSE, const Parameter& para, bool bDualBuffer = false, bool bSaveVolume = false) {
		DISPLAY_FUNCTION;
		try
		{
			//typedef BackprojectionType Backprojection;
			printf("TRANPOSE=%d,bDualBuffer=%d,bSaveVolume=%d\n", TRANPOSE, bDualBuffer ? 1 : 0, bSaveVolume ? 1 : 0);

			ProjectionImageT<float> proj;	
			std::vector<Point2dT<int>> vecIJ(para.ny*para.nx);
	
			for (int i = 0, s = 0; i < para.ny; i++)
			{
				for (int j = 0; j < para.nx; j++, s++)
				{
					vecIJ[s] = Point2dT<int>(j, i);
				}
			}	
	
			VERIFY_TRUE(!vecIJ.empty());
			int nproj = cudaBackProjection::MAX_PROJ;
			VERIFY_TRUE(proj.LoadFolder<float>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj, para.step));
			if (TRANPOSE == 1) 
				proj.Transpose();
	
			std::shared_ptr<BackprojectionType> pBp;
			if (typeid(BackprojectionType) == typeid(BackProjectionRTK) || typeid(BackprojectionType) == typeid(BackProjectionRevRTK)
				|| typeid(BackprojectionType) == typeid(BackProjectionTextureRevRTK))
			{
				pBp = std::make_shared<BackprojectionType>(proj.width, proj.height, cudaBackProjection::MAX_PROJ, para.nx, para.ny, para.nz, bDualBuffer);
			}
			else
			{
				pBp = std::make_shared<BackprojectionType>(para.nz, proj.width, proj.height, cudaBackProjection::MAX_PROJ, (int*)&vecIJ[0], vecIJ.size(), bDualBuffer);		
			}
			//BackprojectionType bp(para.nz, proj.width, proj.height, cudaBackProjection::MAX_PROJ, (int*)&vecIJ[0], vecIJ.size());
			BackprojectionType& bp = *pBp;
		
			std::cout << "begin watch..." << std::endl;
			StopWatchWin watch;
			watch.start();
			//int i = 0;
			for(int i = 0 ; i < para.nProj ; i += cudaBackProjection::MAX_PROJ)
			{
				//std::cout << "bp : " << i << std::endl;
				if(i + cudaBackProjection::MAX_PROJ > para.nProj) break;
				float* pProjData = proj.GetFrame(i).buffer;
				float* pProjMat  = NoConstant(&para.vec_rot_mat_ijk3x4[0] + 12*i);
				bp.BP(TRANPOSE, pProjData, proj.width, proj.height, pProjMat, cudaBackProjection::MAX_PROJ);
			}
			watch.stop();
			std::cout << "end watch..." << std::endl;
			float ftm = watch.getTime();
			float gups = (para.nx / 1000.0)*(para.ny / 1000.0)*(para.nz / 1000.0)*para.nProj / (bp.fKernelTime / 1000.0);
			printf("tranpose=%d, bp time = %f ms, %ffps, kerenl_time=%f, gpus=%f, img_cnt=%d\n", TRANPOSE, ftm, para.nProj / (ftm / 1000), bp.fKernelTime, gups, para.nProj);
			VolumeT<float> vol;
			vol.MallocBuffer(para.nx, para.ny, para.nz);
			bp.GetVolumeData(vol.buffer, vol.nx*vol.ny*vol.nz);
			if (bSaveVolume) 
				VERIFY_TRUE(vol.Save(para.str_working_folder + StringFormat("%d_%d_%d.vol", vol.nx, vol.ny, vol.nz)));
			std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
		}
		catch (std::string str){
			std::cout<<"Exception : "<<str<<std::endl;
		}
		catch (...){
			std::cout << "Exception : " << std::endl;
		}
	}

void TestFDK(int TRANPOSE, const Parameter& para, bool bDualBuffer = false, bool bSaveVolume = false)
{
	DISPLAY_FUNCTION;
	
	typedef BackProjectionFDK BackprojectionType;
	printf("TRANPOSE=%d,bDualBuffer=%d,bSaveVolume=%d\n", TRANPOSE, bDualBuffer ? 1 : 0, bSaveVolume ? 1 : 0);

	ProjectionImageT<float> proj;	
	int nproj = cudaBackProjection::MAX_PROJ;
#ifdef _DEBUG
	VERIFY_TRUE(proj.MallocBuffer(para.nu, para.nv, para.nProj).IsValid());
#else
	VERIFY_TRUE(proj.LoadFolder<float>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj));
#endif
	if (TRANPOSE == 1) 
		proj.Transpose();	
#if 0
	proj.SaveToFolder("C:/tmp/flt-transpose/");
#endif
	std::shared_ptr<BackprojectionType> pBp;

	//geo::BoxT<size_t> subTop(0, 0, 0, para.nx, para.ny/2, para.nz / 2);
	//geo::BoxT<size_t> subBottom(0, 0, para.nz / 2, subTop.width, subTop.height, subTop.depth);

	geo::BoxT<size_t> subTop(0, 0, 0, para.nx, para.ny, para.nz / 4);
	//geo::BoxT<size_t> subBottom(0, 0, para.nz *3/ 4, subTop.width, subTop.height, subTop.depth);
	geo::BoxT<size_t> subBottom(0, 0, para.nz * 3 / 4, subTop.width, subTop.height, 0);
	
	//geo::BoxT<size_t> subTop(para.nx / 4, para.ny / 4, 0, para.nx / 2, para.ny / 2, para.nz / 2);
	//geo::BoxT<size_t> subBottom(subTop.MinX(), subTop.MinY(), para.nz * 1 / 2, subTop.width, subTop.height, subTop.depth);
	
	
	pBp = std::make_shared<BackprojectionType>(proj.width, proj.height, cudaBackProjection::MAX_PROJ, para.nx, para.ny, para.nz, subTop, subBottom, bDualBuffer);

	//BackprojectionType bp(para.nz, proj.width, proj.height, cudaBackProjection::MAX_PROJ, (int*)&vecIJ[0], vecIJ.size());
	BackprojectionType& bp = *pBp;
		
	std::cout << "begin watch..." << std::endl;
	StopWatchWin watch;
	watch.start();
	//int i = 0;
#ifdef _DEBUG
	for(int i = 0 ; i < cudaBackProjection::MAX_PROJ ; i += cudaBackProjection::MAX_PROJ) 
#else
	for(int i = 0 ; i < para.nProj ; i += cudaBackProjection::MAX_PROJ) 	
#endif
	{
		//std::cout << "bp : " << i << std::endl;
		if(i + cudaBackProjection::MAX_PROJ > para.nProj) break;
		float* pProjData = proj.GetFrame(i).buffer;
		float* pProjMat  = NoConstant(&para.vec_rot_mat_ijk3x4[0] + 12*i);
		{
			//std::string str = "proj-mat : ";
			//for (int s = 0; s < 12; s++) str += StringFormat("%f,", pProjMat[s]);
			//printf("%s\n", str.c_str());			
		}
		bp.BP(TRANPOSE, pProjData, proj.width, proj.height, pProjMat, cudaBackProjection::MAX_PROJ);
	}
	watch.stop();
	std::cout << "end watch..." << std::endl;
	float ftm = watch.getTime();
	printf("tranpose=%d, bp time = %f ms, %ffps\n", TRANPOSE, ftm, para.nProj / (ftm / 1000));
	VolumeT<float> vol;
	vol.MallocBuffer(subTop.depth + subBottom.depth, subTop.height, subTop.width);
	bp.GetVolumeData(vol.buffer, vol.BufferSize());
	if (bSaveVolume) 
		VERIFY_TRUE(vol.Save(para.str_working_folder + StringFormat("%d_%d_%d.vol.raw", subTop.width, subTop.height, subTop.depth * 2)));
	std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;		
}

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
	para.step = step == 0 ? TOTAL_PROJ / nProj : step;
	//para.nProj /= 2; para.step *= 2;
	
	double fx = double(nx) / para.nx; 
	double fy = double(ny) / para.ny;
	double fz = double(nz) / para.nz;
	para.nx *= fx; para.ny *= fy; para.dx /= fx; para.dy /= fy; para.nz *= fz; para.dz /= fz;
			
	para.init();
	para.str_proj_img_folder = "../../data/proj_img_filtered608x616/";
	//para.str_proj_img_folder = "../../data/phantom3d/filtered-shepp-logan_w2048_h2048_c1200/";
	//para.str_proj_img_folder = sSrcDir + StringFormat("shepp-logan_w%d_h%d_c%d/", nu, nv, TOTAL_PROJ);
	VERIFY_TRUE(IsPathExisted(para.str_proj_img_folder.c_str()));
	para.str_working_folder = sDstDir;
	//para.Display();
	return para;
}


int main(int argc, char *argv[])
{
	DISPLAY_FUNCTION;
	std::string str;
	printf("argc=%d : ", argc);
	for (int i = 0; i < argc; i++) {
		str += StringFormat("%s ", argv[i]);
	}
	printf("Main_CMD : %s\n", str.c_str());
	VERIFY_TRUE(argc == 11);
	
	int nu, nv;
	int projs, total_proj;
	int nx, ny, nz;
	int iSave = 0;
	std::string src_dir, dst_dir;
	nu = atoi(argv[1]);
	nv = atoi(argv[2]);
	projs = atoi(argv[3]);
	nx = atoi(argv[4]);
	ny = atoi(argv[5]);
	nz = atoi(argv[6]);
	total_proj = atoi(argv[7]);
	src_dir = argv[8];
	dst_dir = argv[9];	
	iSave = atoi(argv[10]);	
		
	const Parameter& para = GetDigitalPhantom(nx, ny, nz, nu, nv, projs, src_dir, dst_dir, total_proj);
	para.Display();
	bool bSave = iSave == 0 ? false : true;
	
	TestFDK(1, para, false, true); return 0;
	
	Test<BackProjectionRTK>(1, para, false, bSave);


	//Test<BackProjectionTextureRevRTK>(0, para, false, bSave);	
	//Test<BackProjectionTextureRevRTK>(0, para, true, bSave);
	//Test<BackProjectionTextureRevRTK>(1, para, false, bSave);
	//Test<BackProjectionTextureRevRTK>(1, para, true, bSave);	
	
	//Test<BackProjectionRevRTK>(0, para, false, bSave);	
	//Test<BackProjectionRevRTK>(1, para, false, bSave);	
	
	//Test<BackProjectionGlobalmem>(0, para, false, bSave);
	//Test<BackProjectionGlobalmem>(1, para, false, bSave);
	//Test<BackProjectionRTK>(0, para, false, bSave);
	//Test<BackProjectionGlobalmem>(1, para, false, bSave);
	//Test<BackProjectionGlobalmem>(1, para, true, false);
	//Test<BackProjectionTexture>(0, para, false, bSave);
	//Test<BackProjectionTexture>(1, para, false, bSave);
	if (0) {
		//Test<BackProjectionRTK>(0, para, true);
		//Test<BackProjectionGlobalmem>(0, para, false);
		Test<BackProjectionGlobalmem>(1, para, true);
		//Test<BackProjectionTexture>(0, para, false);
		//Test<BackProjectionTexture>(1, para, false);		
	}
	
	if (0) {
		Test<BackProjectionRTK>(0, para, false, false);
		//Test<BackProjectionGlobalmem>(0, para, false, false);
		Test<BackProjectionGlobalmem>(1, para, false, false);
		Test<BackProjectionTexture>(0, para, false, false);
		Test<BackProjectionTexture>(1, para, false, false);	
	}
	return 0;
}
