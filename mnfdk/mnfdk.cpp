#include "MpiWrapper.h"




Parameter& Get608x616x512(
	std::string sSrcDir = "../../data/proj-img/", 
	std::string sDstDir = "../../working/flt-proj-img/", 
	int nx = 512, int ny = 512, int nz = 512)
{
	static std::shared_ptr<Parameter> pPara;
	pPara = std::make_shared<Parameter>();
	Parameter& para = *pPara;
	para.nu = 608; para.nv = 616; para.nProj = 512/1;
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


struct TestThread : public MyThread {
	TestThread()
		: MyThread() {
	}
	virtual void OnProc() {
		for (; !m_bExit;)
		{
			std::cout << "OnProc Begin, threadId = " << id() << " ......" << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	
		std::cout << "OnProc End  , threadId = " << id() << " ......" << std::endl;
	}
};

void Test()
{	
	if (1){
		Parameter& para = Get608x616x512();
		const int MAX_COUNT = 32;
		BufferT<float> projImgs, projMats;	
		const int PROJ_IMG_SIZE = para.nu*para.nv;
		const int PROJ_MAT_SIZE = 12;
		projImgs.malloc(PROJ_IMG_SIZE*MAX_COUNT);
		projMats.malloc(PROJ_MAT_SIZE*MAX_COUNT);
		VERIFY_TRUE(projImgs.data() && projMats.data());
	}
	if (0) {
		Parameter& para = Get608x616x512();
		Buffering buf0(para, 16, 9, 0);
		Buffering buf1(para, 16, 9, 1);
		Buffering buf2(para, 16, 9, 5);		
	}
	if (0)
	{
		MyThread thd;
		bool b = thd.IsActive();
		b = thd.IsActive();
	}

	if (0){
		TestThread trd;
		trd.Create();
		trd.Destroy();		
	}	

	if (0){
		SHM server(SHM::SERVER, 1024 * 1024); strcpy(server.buffer, "hello, client");
		SHM client(SHM::CLIENT, 1024 * 1024); std::cout << client.buffer << std::endl;			
	}
	
	if (0)
	{
		SafeList<int*> lst;
		for (int i = 1; i < 11; i++){		
			lst.push_back((int*)i);
		}
		int* p = 0;
		lst.get_front(&p); std::cout << "p="<<p<<std::endl;
		lst.get_front(&p); std::cout << "p=" << p << std::endl;
		lst.get_front(&p); std::cout << "p=" << p << std::endl;
	}

}


int main(int argv, char* argc[]) {
	//Test();
	//return 0;
	Parameter& para = Get2048x2048x1200();
	//Parameter& para = Get608x616x512();
	const int group = 9;
	int rank, size, i;
	system(StringFormat("./clearSMEM.sh").c_str());
	MPI_Status state;
	StopWatchWin watch;
	watch.start();
	{
		MPI_Init(&argv, &argc);
		MpiWraper mpi(para, group);
		mpi.Run();
	
		MPI_Finalize();
	}
	watch.stop();
	std::cout << "total time : " << watch.getAverageTime() << std::endl;
	return 0;
}

void tt()
{
	//	if (rank == 0) {
	//		for (i = 1; i < size; i++) {
	//			sprintf(msg, "hello, %d, this is zero, I'am your master", i);
	//			MPI_Send(msg, 128, MPI_CHAR, i, 0, MPI_COMM_WORLD);
	//		}
	//		for (i = 1; i < size; i++) {
	//			MPI_Recv(rev, 128, MPI_CHAR, i, 0, MPI_COMM_WORLD, &state);
	//			printf("P%d got: %s\n", rank, rev);
	//		}
	//	} else {
	//		MPI_Recv(rev, 128, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &state);
	//		printf("P%d got: %s\n", rank, rev);
	//		sprintf(msg, "hello, zero, this is %d, I'am your slave", rank);
	//		MPI_Send(msg, 128, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	//	}	
}