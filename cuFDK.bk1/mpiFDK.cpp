#include "mpiFDK.h"
#include "../common/cudaLib.cuh"
#include "../Filterlib/FilterLib.h"
//////////////////////////////////////////////////////////////////////////////////////
template<typename T>
inline T PopDataFromList(SafeList<T>& lst, bool* pDone = NULL) {
	T pHeader = NULL;
	for (;;) {
		lst.get_front(&pHeader);
		if (pHeader) break;
		if (pDone){
			if (*pDone) {
				if (lst.size() == 0)
					break;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return pHeader;
}

template<typename T>
	inline SafeList<T>& PushDataToList(SafeList<T>& lst, T& pHeader) {
		if (pHeader){
			lst.push_back(pHeader);			
		}
		return lst;
	}
/////////////////////////////////////////////////////////////
FilterThread::FilterThread(mpiFDK* _pMPI) : pMPI(_pMPI), bDone(false)
{
	VERIFY_TRUE(pMPI);
	imgSize = pMPI->para.nu*pMPI->para.nv*sizeof(float);
	blkSize = imgSize + sizeof(DataHeader);
	totalSize = blkSize*MAX_IMG_COUNT;	
	VERIFY_TRUE(buffer.malloc(totalSize));
	char* p = buffer.data();
#pragma omp parall for num_threads(32)
	for (int i = 0; i < MAX_IMG_COUNT; i++, p += blkSize){		
		memset(p, 0, sizeof(DataHeader));
		DataHeader* _p = (DataHeader*)p;
		_p->id = -1;
		_p->dim[0] = pMPI->para.nu;
		_p->dim[1] = pMPI->para.nv;
		_p->head_size = sizeof(DataHeader);
		_p->total_size = blkSize;
		lstEmpty.push_back(_p);		
	}
	VERIFY_TRUE(lstEmpty.size()>0);
}

void FilterThread::OnProc()
{
	VERIFY_TRUE(lstEmpty.size() > 0);
	char name[128] = "";
	//gpuDevice device(pMPI->color);
	//strcpy(name, device.prop.name);
	const Parameter& para = pMPI->para;
	blk_per_rank = UpDivide(para.nProj, pMPI->mpi2d.row.size*pMPI->mpi2d.col.size);
	blk_per_comm = blk_per_rank*pMPI->mpi2d.col.size;

	int start_idx = blk_per_comm*pMPI->mpi2d.row.rank + blk_per_rank*pMPI->mpi2d.col.rank;
	int end_idx = MIN(start_idx + blk_per_rank, para.nProj);
	printf("xy(%d, %d), start_idx=%d, end_idx=%d, gpu=%s\n", pMPI->GetXY().x, pMPI->GetXY().y, start_idx, end_idx, name);
	
#pragma omp parallel for num_threads(32)	
	for (int i = start_idx; i < end_idx; i++){
		ImageT<float> img(para.nu, para.nv);
		char szPath[1024];
		sprintf(szPath, "%simg%04d.raw", para.str_proj_img_folder.c_str(), i);
		VERIFY_TRUE(FileSize(szPath) == sizeof(float)*para.nu*para.nv);
		DataHeader* pHeader = PopDataFromList(lstEmpty);
		VERIFY_TRUE(pHeader);
		VERIFY_TRUE(LoadFromFile(szPath, (float*)img.buffer, para.nu*para.nv));
		{
			
			FrameT<float> src((float*)img.buffer, para.nu, para.nv, 0);
			FrameT<float> dst((float*)pHeader->data, para.nu, para.nv, 0);
			FilterEngine flt(para.cos_image_table.cos_sigma.GetFrame(), para.fft_kernel_real_f32);
			flt.Filtering(src, dst);
		}
		pHeader->id = i;
		lstReady.push_back(pHeader);
	}
	for (int i = end_idx; i < start_idx + blk_per_rank; i++){
		DataHeader* pHeader = PopDataFromList(lstEmpty);
		VERIFY_TRUE(pHeader);
		pHeader->id = -1;
		lstReady.push_back(pHeader);
	}
	bDone = true;
}

//////////////////////////////////////////////////////////////////////////////////////
BpThread::BpThread(mpiFDK* _pMPI) : pMPI(_pMPI)
{
	DISPLAY_FUNCTION;
}

BpThread::~BpThread()
{
	DISPLAY_FUNCTION;
}

void BpThread::OnProc()
{
	cudaSetDevice(pMPI->mpi2d.rank % pMPI->ranks_per_node);
	const Parameter& para = pMPI->para;	
	int col_size = pMPI->mpi2d.col.size;
	int count = 0;
	int blkSize = para.nu*para.nv*sizeof(float) + sizeof(DataHeader);
	int blk_per_rank = pMPI->threadFlt->blk_per_rank;
	int blk_per_comm = blk_per_rank*col_size;
	//printf("blk_per_rank=%d, blk_per_comm=%d\n", blk_per_rank, blk_per_comm);

	for (;;){
		//char* buf = PopDataFromList(pMPI->lstReady, &pMPI->threadFlt->bDone);
		char* buf = PopDataFromList(pMPI->lstReady);
		if (!buf){
			printf("error : buf == 0\n");
			break;
		}
		char* p = buf;	
		for (int i = 0; i < col_size; i++, p += blkSize) {
			DataHeader* pHeader = (DataHeader*)p;
			VERIFY_TRUE(pHeader->total_size == blkSize);
			if (pHeader->id >= 0){	
				//WriteToFile<char, char>(StringFormat("%s/img%04d.raw", "/home/chen/dump", pHeader->id), pHeader->data, para.nu*para.nv*sizeof(float));
			}
		}	
		count += col_size;
		//printf("count = %d\n", count);
		PushDataToList(pMPI->lstEmpty, buf);
		if (count >= blk_per_comm) {
			printf("bp done\n");
			break;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////

mpiFDK::mpiFDK(const Parameter& _para, int _ranks_per_node = 2)
	: para(_para)
	, ranks_per_node(_ranks_per_node)
{
	DISPLAY_FUNCTION;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	mpi2d.build(world_rank, world_size, world_size / ranks_per_node, ranks_per_node);
	
	printf("hostname=%s, world_rank=%d, world_size=%d, ranks_per_node=%d, x=%d, y=%d, pid=%d\n", 
		hostname().c_str(),
		world_rank,
		world_size,
		GetXY().x,
		GetXY().y,
		mpi2d.col.rank,
		getpid());
#if DEBUG==1
	const char* dbgfile = "__dbg.txt";
	DeleteFile(dbgfile);
	while (!IsPathExisted(dbgfile)) sleep(1);
	std::cout << "dbg.txt ok!" << std::endl;
#endif
	threadFlt = std::make_shared<FilterThread>(this);
	threadFlt->Create();
	{
		const int BATCH_COUNT = 32;
		max_multi_data_count = UpDivide(MAX(BATCH_COUNT * 2, mpi2d.col.size), mpi2d.col.size);
		max_multi_data_count = MAX(max_multi_data_count, 2);
		int64 blkSize = threadFlt->blkSize*mpi2d.col.size;
		int64 totalSize = blkSize*max_multi_data_count;
		buffer.malloc(totalSize);
		char* p = buffer.data();
		for (int i = 0; i < max_multi_data_count; i++, p += blkSize) {
			this->lstEmpty.push_back(p);
		}
	}
	threadBp = std::make_shared<BpThread>(this);
	threadBp->Create();
}


mpiFDK::~mpiFDK()
{
	DISPLAY_FUNCTION;
	threadFlt.reset();
	threadBp.reset();
}

Point2dT<int> mpiFDK::GetXY() const
{
	return Point2dT<int>(mpi2d.row.rank, mpi2d.col.rank);
}

bool mpiFDK::Run()
{
	VERIFY_TRUE(threadFlt);
	
	//printf("begin\n");
	for (int nAllGatherCount = 0; ;) {
		if (threadFlt->bDone) {
			if (threadFlt->lstReady.size() == 0) 
				break;			
		}
		DataHeader* pHeader = PopDataFromList(threadFlt->lstReady, &threadFlt->bDone);
		if (!pHeader) {
			printf("Error : pHeader == NULL\n");
		}
#if 1	
		char* rcvBuf = PopDataFromList(lstEmpty);
		VERIFY_TRUE(rcvBuf);
		MPI_Request request;
		MPI_Status state;
		
		const int size = para.nu*para.nv*sizeof(float) + sizeof(*pHeader);
		VERIFY_TRUE(pHeader->total_size == size);
#if 0
		MPI_Iallgather(pHeader, size, MPI_CHAR, rcvBuf, size, MPI_CHAR, mpi2d.col.comm, &request);
		//printf("MPI_Wait    ----begin\n");
		int ret = MPI_Wait(&request, &state);
		//printf("MPI_Wait    ----end\n");
		if (ret != MPI_SUCCESS) {
			std::cout << "error:MPI_Wait" << ret << std::endl;
			MPI_Abort(mpi2d.col.comm, -1000);
		}	
#else
		MPI_Allgather(pHeader, size, MPI_CHAR, rcvBuf, size, MPI_CHAR, mpi2d.col.comm);
#endif
		PushDataToList(lstReady, rcvBuf);
#endif
		PushDataToList(threadFlt->lstEmpty, pHeader);

		nAllGatherCount++;
		if (nAllGatherCount > threadFlt->blk_per_rank)
			break;
	}
	//printf("end\n");
	while(!threadFlt->bDone) std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
}