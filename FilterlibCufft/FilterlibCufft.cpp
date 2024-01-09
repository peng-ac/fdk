#include <iostream>
#include "../common/type.h"


using namespace std;


int test_filter(const Parameter& para, int batchCount)
{
	extern float FilterProjImage(const Parameter& para, const ProjectionImage& proj_img, int DEPTH = 32);
	//////////////////////////////////////////////////
	const char* pSrc = para.str_proj_img_folder.c_str();
	const char* pDst = para.str_working_folder.c_str();
	int order = log(para.fft_kernel_real.size()) / log(2) + 0.5;
	ProjectionImageT<float> proj;
	VERIFY_TRUE(proj.LoadFolder<float>(pSrc, para.nu, para.nv, para.nProj, para.step));
	float tm = FilterProjImage(para, proj, batchCount);
	printf("filtering_%dx%d, count=%d, time=%f ms, avg=%f ms, %f fps\n", para.nu, para.nv, para.nProj, tm, tm / para.nProj, para.nProj / (tm/1000.f));
	proj.SaveToFolder(pDst);
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
	//para.str_proj_img_folder = "../../data/phantom3d/filtered-shepp-logan_w2048_h2048_c1200/";
	para.str_proj_img_folder = sSrcDir + StringFormat("shepp-logan_w%d_h%d_c%d/", nu, nv, TOTAL_PROJ);
	VERIFY_TRUE(IsPathExisted(para.str_proj_img_folder.c_str()));
	para.str_working_folder = sDstDir;
	//para.Display();
	return para;
}

int main(int argc, char **argv)
{
	DISPLAY_FUNCTION;
	std::string str;
	printf("argc=%d : ", argc);
	for (int i = 0; i < argc; i++) {
		str += StringFormat("%s ", argv[i]);
	}
	printf("Main_CMD : %s\n", str.c_str());
	VERIFY_TRUE(argc == 7);
	
	int nu, nv;
	int projs;
	int batchCount = 32;
	std::string src_dir, dst_dir, threads("1");
	nu = atoi(argv[1]);
	nv = atoi(argv[2]);
	projs = atoi(argv[3]);
	src_dir = argv[4];
	dst_dir = argv[5];	
	batchCount = atoi(argv[6]);	 
		
	Parameter& para = GetDigitalPhantom(512, 512, 512, nu, nv, projs, src_dir, dst_dir, 16384, 1);
	para.Display();
	
	std::cout << "batch count = " << batchCount << std::endl;
	test_filter(para, batchCount);
	std::cout << "" << std::endl;
	return 0;
}