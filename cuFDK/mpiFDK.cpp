#include "mpiFDK.h"
#include "../common/cudaLib.cuh"
#include "../Filterlib/FilterLib.h"
#include <omp.h>

//////////////////////////////////////////////////////////////////////////////////////
template<typename T>
inline T PopDataFromList(SafeList<T>& lst, bool* pDone = NULL) {
	T pHeader = NULL;
	for (uint i=0;;i++) {
		lst.get_front(&pHeader);
		if (pHeader) break;
		if (pDone){
			if (*pDone) {
				if (lst.size() == 0)
					break;
			}
		}
		if ((i+1) % 100000 == 0) printf("*");
		std::this_thread::sleep_for(std::chrono::microseconds(10));
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
FilterThread::FilterThread(mpiFDK* _pMPI) : pMPI(_pMPI)
{
	VERIFY_TRUE(pMPI);
	Info& info = pMPI->info;
	const size_t MAX_IMG_COUNT = MIN(2048,pMPI->para.nProj/pMPI->mpi2d.size);

	info.imgSize = pMPI->para.nu*pMPI->para.nv*sizeof(float);
	info.blkSize = info.imgSize + sizeof(DataHeader);
	info.totalSize = info.blkSize*MAX_IMG_COUNT;	
	VERIFY_TRUE(buffer.malloc(info.totalSize));
	char* p = buffer.data();
	for (int i = 0; i < MAX_IMG_COUNT; i++, p += info.blkSize) {		
		memset(p, 0, sizeof(DataHeader));
		DataHeader* _p = (DataHeader*)p;
		_p->id = -1;
		_p->dim[0] = pMPI->para.nu;
		_p->dim[1] = pMPI->para.nv;
		_p->head_size = sizeof(DataHeader);
		_p->total_size = info.blkSize;
		strcpy(_p->name, "header");
		lstEmpty.push_back(_p);		
	}
	VERIFY_TRUE(lstEmpty.size()>0);
}

void FilterThread::OnProc()
{
	VERIFY_TRUE(lstEmpty.size() > 0);
	Info& info = pMPI->info;
	const Parameter& para = pMPI->para;
	char name[128] = "";
	pMPI->initInfo();
	
	float t1, t2;
	StopWatchWin watch;
	watch.start();
	unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
	pMPI->logInfo.cpuCores = concurentThreadsSupported;
	//int nProcessor = omp_get_num_procs();
	int omp_threads = MIN(concurentThreadsSupported / FilterEngine::THREAD_COUNT / pMPI->ranks_per_node - 3, info.blk_per_rank);
	omp_threads = MAX(1, omp_threads);
	//omp_threads = 80;
	omp_set_num_threads(omp_threads);
	std::vector<std::shared_ptr<FilterEngine>> vecFlt(omp_threads);
	watch.stop();
	t1 = watch.getAverageTime();
	pMPI->logInfo.fFilterInitTime = t1;
	//	#pragma omp parallel for 
	//	for (int i = 0; i < omp_threads; i++){
	//		vecFlt[i] = std::make_shared<FilterEngine>(para.cos_image_table.cos_sigma.GetFrame(), para.fft_kernel_real_f32);
	//	}
	
		//FilterEngine flt(para.cos_image_table.cos_sigma.GetFrame(), para.fft_kernel_real_f32);
		//gpuDevice device(pMPI->color);
		//strcpy(name, device.prop.name);
	float tmLoad = 0;
	float tmFlt = 0;
	MyPrintf("flt-thread->rank=%d,cores=%d, omp_threads=%d, init-time=%f\n", pMPI->mpi2d.rank, concurentThreadsSupported, omp_threads, t1);
	watch.start();
	#pragma omp parallel for num_threads(omp_threads) reduction(+:tmLoad, tmFlt)
	for(int s = 0 ; s < info.blk_per_rank ; s++){
		VERIFY_TRUE(omp_get_num_threads() == omp_threads);
		//printf("flt.thread id = %d\n", omp_get_thread_num());
		StopWatchWin w2, w3;
		int i = info.start_idx + s;
		DataHeader* pHeader = PopDataFromList(lstEmpty);
		VERIFY_TRUE(pHeader);	
		VERIFY_TRUE(pHeader->total_size > 0);
		VERIFY_TRUE(info.end_idx <= para.nProj);
		VERIFY_TRUE(pHeader->total_size == info.blkSize);
		VERIFY_TRUE(strcmp(pHeader->name, "header") == 0);
		if (i < info.end_idx){
			ImageT<float> img(para.nu, para.nv);
			//ImageT<float> _dst(para.nv, para.nu*2);
			char szPath[1024] = "";
			strcpy(szPath, para.GetProjPath(i).c_str());
			//sprintf(szPath, "%simg%04d.raw", para.str_proj_img_folder.c_str(), i);
			//VERIFY_TRUE(FileSize(szPath) == sizeof(float)*para.nu*para.nv);

			w2.start();
			const bool bShowLoad = false;
			VERIFY_TRUE(LoadFromFile(szPath, (float*)img.buffer, para.nu*para.nv, bShowLoad));
			w2.stop();
			tmLoad += w2.getAverageTime();
			{
				pHeader->bTranspose = 1;
				FrameT<float> src((float*)img.buffer, para.nu, para.nv, 0);
				FrameT<float> dst((float*)pHeader->data, para.nv, para.nu, 0);
				int tid = omp_get_thread_num();
				if (!vecFlt[tid])
					vecFlt[tid] = std::make_shared<FilterEngine>(para.cos_image_table.cos_sigma.GetFrame(), para.fft_kernel_real_f32);
				w3.start();
				vecFlt[tid]->Filtering(src, dst, true);
				w3.stop();
				tmFlt += w3.getAverageTime();
			}
			VERIFY_TRUE(i >= 0);
			pHeader->id = i;
			VERIFY_TRUE(pHeader->total_size == info.blkSize);
			VERIFY_TRUE(pHeader->id < para.nProj);	
			//printf("flt (%d), load-time=%fms, flt-time=%fms, path=%s\n", i, w2.getAverageTime(), w3.getAverageTime(), szPath);
		}else{
			memset(pHeader->data, 0, sizeof(float)*para.nu*para.nv);
			pHeader->bTranspose = 1;
			pHeader->id = -1;		
		}
		VERIFY_TRUE(strcmp(pHeader->name, "header") == 0);
		if (pHeader->id >= para.nProj)
			printf("pHeader->id = %d, para.nProj = %d\n", pHeader->id, para.nProj);
		VERIFY_TRUE(pHeader->id < para.nProj);
		VERIFY_TRUE(pHeader->total_size > 0);
		VERIFY_TRUE(pHeader->total_size == info.blkSize);
		lstReady.push_back(pHeader);
	}
	watch.stop();
	t2 = watch.getAverageTime();
	pMPI->logInfo.nLoadImg += info.end_idx - info.start_idx;
	pMPI->logInfo.fLoadFltTime += t2;
	printf("Filtering, rank=%d, ImageCount=%d, omp_threads=%d, Init-time=%f, Load-time=%f, Filtering-time=%f, (flt+load)=%f ms, avg=%f ms\n", pMPI->mpi2d.rank, info.end_idx - info.start_idx, omp_threads, t1, tmLoad, tmFlt, t2, t2 / (info.end_idx - info.start_idx));
	pMPI->bThreadFltDone = true;
	pMPI->logInfo.fThreadFlt = pMPI->logInfo.fFilterInitTime + t2;
}

//////////////////////////////////////////////////////////////////////////////////////
BpThread::BpThread(mpiFDK* _pMPI) : pMPI(_pMPI)
{
	//DISPLAY_FUNCTION;
}

BpThread::~BpThread()
{
	//DISPLAY_FUNCTION;
}

void BpThread::PushBpData(int id, const float* pImg, const float* pMat){
	bool bBp = false;
	const Parameter& para = pMPI->para;		
	if (pImg && pMat) {
		strMsg += StringFormat("%d,", id);
		{
			//std::string str = "proj-mat : ";
			//for (int s = 0; s < 12; s++) str += StringFormat("%f,", pMat[s]);
			//printf("%s\n", str.c_str());			
		}
		memcpy(&projImg.data()[para.nu*para.nv*nNum], pImg, para.nu*para.nv*sizeof(float));
		memcpy(&projMat.data()[cudaBackProjection::PROJ_MAT_SIZE*nNum], pMat, sizeof(float) * cudaBackProjection::PROJ_MAT_SIZE);
		nNum++;
		if (nNum >= cudaBackProjection::MAX_PROJ)
			bBp = true;
	} else if (pImg == NULL && pMat == NULL){
		VERIFY_TRUE(id < 0);
		if (nNum > 0)
			bBp = true;
	}
		
	if (bBp) {
		int device = -1;
		cudaGetDevice(&device);
		VERIFY_TRUE(device == pMPI->mpi2d.rank % pMPI->ranks_per_node);
		//printf("--------begin BP, nNum=%d, w=%d, h=%d, msg=%s\n", nNum, para.nv, para.nu, strMsg.c_str());
		strMsg.clear();
		StopWatchWin watch;
		watch.start();
		pBP->BP(1, projImg.data(), para.nv, para.nu, projMat.data(), nNum);
		watch.stop();
		pMPI->logInfo.fBpTime += watch.getAverageTime();
		MyPrintf("Bp->rank=%d, gpu%d, bp %d imgs, time=%f ms, avg=%f ms\n", pMPI->mpi2d.rank, device, nNum, watch.getAverageTime(), watch.getAverageTime() / nNum);
		//printf("--------end   BP\n");
		nNum = 0;
	}
}

void BpThread::OnProc()
{
	float tmSetDev = 0;
	StopWatchWin tmBp;

	//cudaSetDevice(pMPI->mpi2d.rank % pMPI->ranks_per_node);
	tmBp.start();
	cudaSetDevice(pMPI->mpi2d.rank % pMPI->ranks_per_node);
	tmBp.stop();
	_CUDA_CHECK_ERROR(CUDAString("pMPI->mpi2d.rank=%d, pMPI->ranks_per_node=%d", pMPI->mpi2d.rank, pMPI->ranks_per_node));	
		
	int device = -1;
	cudaGetDevice(&device);
	VERIFY_TRUE(device == pMPI->mpi2d.rank % pMPI->ranks_per_node);
	
	tmSetDev = tmBp.getAverageTime();
	
	tmBp.reset(); tmBp.start();

	const Parameter& para = pMPI->para;	
	Info& info = pMPI->info;
	size_t row_size = pMPI->mpi2d.row.size;
	
	size_t blkSize = info.blkSize;
	size_t blk_per_rank = info.blk_per_rank;
	size_t blk_per_comm = blk_per_rank*row_size;
	MyPrintf("blk_per_rank=%d, blk_per_comm=%d\n", blk_per_rank, blk_per_comm);
	
	size_t _projWidth = para.nv;
	size_t _projHeight = para.nu;
	size_t _projCount = cudaBackProjection::MAX_PROJ;
	size_t _nx = para.nx;
	size_t _ny = para.ny;
	size_t _nz = para.nz;
	geo::BoxT<size_t>& _subTop = pMPI->divVol->subTop;
	geo::BoxT<size_t>& _subBottom = pMPI->divVol->subBottom;
	bool bDualBuffer = false;
	StopWatchWin watch;
	watch.start();
	//Transpose
	pBP = std::make_shared<BackProjectionFDK>(_projWidth, _projHeight, _projCount, _nx, _ny, _nz, _subTop, _subBottom, bDualBuffer);
	watch.stop();
	pMPI->logInfo.fCudaInitTime = watch.getAverageTime();
	
	projImg.malloc(_projWidth*_projHeight*_projCount); 
	projMat.malloc(cudaBackProjection::PROJ_MAT_SIZE*_projCount);
	nNum = 0;

	size_t nPushDataCount = 0;
	const size_t nImgSize = sizeof(float)*para.nu*para.nv;
	for (int count = 0;;) {
		//char* buf = PopDataFromList(pMPI->lstReady, &pMPI->threadFlt->bDone);
		char* buf = PopDataFromList(pMPI->lstReady, &pMPI->bAllGatherDone);
		if (!buf){
			printf("error : buf == 0\n");
			break;
		}

		//std::string str;
		char* p = buf;	
		for (size_t i = 0; i < row_size; i++, p += blkSize) {
			DataHeader* pHeader = (DataHeader*)p;
			//printf("i=%d, id=%d, dim[0]=%d, dim[1]=%d, total_size = %d, blkSize = %d\n", i, pHeader->id, pHeader->dim[0], pHeader->dim[1], pHeader->total_size, blkSize);
			VERIFY_TRUE(pHeader->total_size == blkSize);
			VERIFY_TRUE(pHeader->id < para.nProj);
			VERIFY_TRUE(pHeader->bTranspose == 1);			
			if (pHeader->id >= 0 && pHeader->id < para.nProj){
				if (pMPI->GetXY().x == 0){
					const char* home = GetSaveHome();
					if (IsPathExisted(home)){
						//WriteToFile<char, char>(StringFormat("%sdump/img%04d.raw", home, pHeader->id), pHeader->data, para.nu*para.nv*sizeof(float));
					}
				}
				PushBpData(pHeader->id, (float*)pHeader->data,
					&para.vec_rot_mat_ijk3x4[0] + cudaBackProjection::PROJ_MAT_SIZE*pHeader->id);
				nPushDataCount++;
				//str += StringFormat("%d,", pHeader->id);
			}
		}	
		//printf("id : %s\n", str.c_str());
		count ++;
		//printf("count = %d\n", count);
		PushDataToList(pMPI->lstEmpty, buf);
		if (count >= blk_per_rank) {

			break;
		}
	}
	PushBpData(-1, NULL, NULL);
	MyPrintf("bp done, rank=%d, nPushDataCount=%d, nBpCount=%d\n", pMPI->mpi2d.rank, nPushDataCount, pBP->nBpCount);	
	tmBp.stop();
	
	{
		//get volume from gpu and transpose volume
		geo::BoxT<size_t>& top = pMPI->divVol->subTop;
		StopWatchWin w1;
		w1.start();
		pBP->GetVolumeData(pMPI->divVol->buffer, pMPI->divVol->buffer_size);
		w1.stop();
		pMPI->logInfo.fD2HTime = w1.getAverageTime();
		printf("%d, pBP->GetVolumeData time=%f ms\n", pMPI->mpi2d.rank, w1.getAverageTime());
		const char* home = GetSaveHome();
		if (IsPathExisted(home)){
			//WriteToFile<float, float>(StringFormat("%s/dump/(%d-%d)%d_%d_%d.raw", home, pMPI->GetXY().x, pMPI->GetXY().y, top.width, top.height, top.depth * 2), pMPI->divVol->buffer, pMPI->divVol->buffer_size);		
		}
	
	}
	
	MyPrintf("BpAllTime:rank=%d, total_tm=%f, init-time-%f, setdev-time=%f\n", pMPI->mpi2d.rank, tmBp.getAverageTime(), watch.getAverageTime(), tmSetDev);
	pMPI->bBpDone = true;
	
	pMPI->logInfo.fThreadBp = tmBp.getAverageTime();
}

//////////////////////////////////////////////////////////////////////////////////////

mpiFDK::mpiFDK(const Parameter& _para, LogInfo& _logInfo, int _ranks_per_node, int rows /*= 4*/)
	: para(_para)
	, ranks_per_node(_ranks_per_node)
	, logInfo(_logInfo)
{
	//DISPLAY_FUNCTION;
	bThreadFltDone = false;
	bAllGatherDone = false;
	bBpDone = false;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	//int cols = cols;
	size_t cols = world_size / rows;
		
	mpi2d.build(world_rank, world_size, rows, cols, _ranks_per_node);
	
	this->logInfo.rank = world_rank;
	this->logInfo.size = world_size;
	this->logInfo.rows = rows;
	this->logInfo.cols = cols;
	this->logInfo.x = GetXY().x;
	this->logInfo.y = GetXY().y;
	printf("hostname=%s, world_rank=%d, world_size=%d, ranks_per_node=%d, x=%d, y=%d, pid=%d\n", 
		hostname().c_str(),
		world_rank,
		world_size,
		ranks_per_node,
		GetXY().x,
		GetXY().y,
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
		const size_t MAX_BUFFER_COUNT = 1024;
		const size_t BATCH_COUNT = 32;
		max_multi_data_count = MAX_BUFFER_COUNT / mpi2d.row.size;
		size_t blkRowSize = info.blkSize*mpi2d.row.size;
		size_t totalSize = blkRowSize*max_multi_data_count;
		buffer.malloc(totalSize);
		std::vector<char*> vec;
		char* p = buffer.data();
		for (size_t i = 0; i < max_multi_data_count; i++, p += blkRowSize) {
			vec.push_back(p);
		}
		this->lstEmpty.push_back(vec);
		MyPrintf("init->mpi2d.row.size=%d, max_multi_data_count=%d\n", mpi2d.row.size, max_multi_data_count);
	}		
	{
		threadBp = std::make_shared<BpThread>(this);
		threadBp->name = StringFormat("id=%d", world_rank);		
	}
	{
		divVol = std::make_shared<DivVolume>(para.nx, para.ny, para.nz, mpi2d.row.rank, mpi2d.row.size);
	}
	threadBp->Create();	
}


mpiFDK::~mpiFDK()
{
	//DISPLAY_FUNCTION;	
	threadFlt.reset();	
	threadBp.reset();
}

Info& mpiFDK::initInfo()
{
	info.blk_per_rank = UpDivide(para.nProj, mpi2d.row.size*mpi2d.col.size);
	info.blk_per_comm = info.blk_per_rank*mpi2d.row.size;

	info.start_idx = info.blk_per_comm*mpi2d.col.rank + info.blk_per_rank*mpi2d.row.rank;
	info.end_idx = MIN(info.start_idx + info.blk_per_rank, para.nProj);
	printf("xy(%d, %d), start_idx=%d, end_idx=%d\n", GetXY().x, GetXY().y, info.start_idx, info.end_idx);
	
	return info;
}


Point2dT<int> mpiFDK::GetXY() const
{
	return Point2dT<int>(mpi2d.row.rank, mpi2d.col.rank);
}

bool mpiFDK::Run()
{
	//DISPLAY_FUNCTION;
	VERIFY_TRUE(threadFlt);
	VERIFY_TRUE(threadBp);
	
	//printf("begin\n");
	float tmCommunication = 0;
	StopWatchWin com;	
//	BufferT<char> bufCol;
	com.start();
#if 1
	for (int nAllGatherCount = 0; ;) {
		if (bThreadFltDone) {
			if (threadFlt->lstReady.size() == 0) 
				break;			
		}
		double tPopDataFlt = 0, tPopGather = 0; 
		tPopDataFlt = MPI_Wtime();
		//printf("begin PopDataFromList\n");
		VERIFY_TRUE(threadFlt);
		DataHeader* pHeader = PopDataFromList(threadFlt->lstReady, &bThreadFltDone);
		if (!pHeader) {
			printf("Error : pHeader == NULL, nAllGatherCount=%d\n", nAllGatherCount);
			break;
		}
		tPopGather = MPI_Wtime();
#if 1	
		char* rcvBuf = PopDataFromList(lstEmpty);
		VERIFY_TRUE(rcvBuf);
		
		const size_t size = para.nu*para.nv*sizeof(float) + sizeof(*pHeader);
		if (pHeader->total_size != size){
			printf("error : pHeader->total_size=%d, size=%d, id=%d\n", pHeader->total_size, size);
		}
		VERIFY_TRUE(pHeader->total_size == size);
		//MPI_Barrier(mpi2d.row.comm);
		if (mpi2d.row.size > 1){
			MPI_Barrier(mpi2d.row.comm);
			double t1; 
			t1 = MPI_Wtime();
			MPI_Allgather(pHeader, size, MPI_CHAR, rcvBuf, size, MPI_CHAR, mpi2d.row.comm);
			MPI_Barrier(mpi2d.row.comm);
			t1 = MPI_Wtime() - t1;
			t1 *= 1000.f;
			this->logInfo.fAllGatherTime += t1;
			MyPrintf("MPI_Allgather (%d), world_rank=%d, size=%d, time=%f ms, avg=%.3fms/frm\n", nAllGatherCount, mpi2d.rank, mpi2d.row.size, t1, t1 / mpi2d.row.size);			
		}else{
			memcpy(rcvBuf, pHeader, pHeader->total_size);
		}
	
		for (size_t i = 0; i < mpi2d.row.size; i++) {
			char* p = rcvBuf + ((DataHeader*)rcvBuf)->total_size*i;
			VERIFY_TRUE(((DataHeader*)p)->total_size > 0);
		}
		PushDataToList(lstReady, rcvBuf);
#endif
		PushDataToList(threadFlt->lstEmpty, pHeader);

		nAllGatherCount++;
		//printf("threadFlt->blk_per_rank=%d\n", threadFlt->blk_per_rank);
		if (nAllGatherCount >= info.blk_per_rank)
			break;
	}
#endif
	//printf("end\n");
	bAllGatherDone = true;
	com.stop();


	while (1) {
		if (bThreadFltDone && bBpDone) break;
		std::this_thread::sleep_for(std::chrono::microseconds(10)); 
	}
	{
		TwoSubVolume* pSub = NULL;
		float* pRcvBuf = NULL;
		BufferT<float> _buf;
		if (GetXY().y == 0){
			_buf.malloc(divVol->buffer_size + sizeof(TwoSubVolume));
			pSub = (TwoSubVolume*)_buf.data();
			pSub->subTop = divVol->subTop;
			pSub->subBottom = divVol->subBottom;
			pRcvBuf = pSub->data;
		}
		
		
#if 1
		if (mpi2d.col.size > 1){
			//printf("begin MPI_Reduce, rank=%d\n", mpi2d.rank);
			size_t repeat = 1;
			size_t extra  = 0;
			double t1;		
			//MPI_Barrier(MPI_COMM_WORLD);		
			for (size_t k = 0; k < repeat + extra; k++) {
				if (k >= extra) {
					MPI_Barrier(MPI_COMM_WORLD);
					t1 = MPI_Wtime();
				}
				const int BATCH_SIZE = INT_MAX;
				//const int BATCH_SIZE = 1024 * 1024 * 16;
				for (size_t t = 0; t < divVol->buffer_size; t += BATCH_SIZE) {
					auto pSnd = divVol->buffer + t;
					auto pRcv = pRcvBuf + t;
					size_t count = MIN(BATCH_SIZE, divVol->buffer_size - t);
					MPI_Reduce(pSnd, pRcv, count, MPI_FLOAT, MPI_SUM, 0, mpi2d.col.comm);
					//MPI_Reduce(divVol->buffer, pRcvBuf, divVol->buffer_size, MPI_REAL, MPI_SUM, 0, mpi2d.col.comm);				
				}				
			}
			MPI_Barrier(MPI_COMM_WORLD);
			t1 = MPI_Wtime() - t1; 
			t1 *= 1000;
			t1 /= double(repeat);
			logInfo.fReduceTime = t1;
			//std::cout << "### divVol->buffer_size = " << divVol->buffer_size << std::endl;
			MyPrintf("MPI_Reduce, world_rank=%d, size=%d, time=%f ms, data_size=%f GB, %.3f GB/s\n", mpi2d.rank, mpi2d.col.size, t1, (float)ToGB(divVol->buffer_size) * 4, (float)ToGB(divVol->buffer_size) * 4. / (t1 / 1000.));			
		} else 	{
			pRcvBuf = this->divVol->buffer;
		}
#else
		pRcvBuf = this->divVol->buffer;
#endif	
		if (GetXY().y == 0) {
			geo::BoxT<size_t>& top = divVol->subTop;
			const char* home = GetSaveHome();
			VERIFY_TRUE(IsPathExisted(home));
			{
				MPI_Barrier(mpi2d.row.comm);				
				double t1; 
				t1 = MPI_Wtime();
#if 0
				WriteToFile<float, float>(StringFormat("%sdump/reduce.(%d-%d)%d_%d_%d.raw", home, GetXY().x, GetXY().y, top.width, top.height, top.depth * 2), pRcvBuf, divVol->buffer_size);	
#endif
				VERIFY_TRUE(divVol->buffer);
				size_t slice_size = divVol->nx*divVol->ny;
				VERIFY_TRUE(divVol->nx == divVol->subTop.width);
				VERIFY_TRUE(divVol->ny == divVol->subTop.height);
	
				for (size_t j = 0; j < 2; j++) {
					geo::BoxT<size_t>* pBox = &divVol->subTop;
					float* pBuf = pRcvBuf;		
					if (j == 1) {
						pBox = &divVol->subBottom;
						pBuf += divVol->subTop.depth*slice_size;
					}
					if (pBox->depth > 0){
						omp_set_num_threads(MAX(1, pBox->depth));
						#pragma omp parallel for
						for (size_t k = 0; k < pBox->depth; k++) {
							size_t idx = pBox->z + k;
							std::string str = StringFormat("%sdump/reduce-%d_%d_%d-%d_%d/%06d-(%d_%d)(%d_%d_%d).raw", 
								home,
								divVol->nx,
								divVol->ny,
								divVol->nz,	 
								this->mpi2d.row.size,
								mpi2d.col.size,
								idx,
								GetXY().x,
								GetXY().y,
								top.width,
								top.height,
								top.depth * 2);
							auto WTF = WriteToFile<float, float>;
							VERIFY_TRUE(WTF(str, pBuf + k*slice_size, slice_size, false));
						}						
					}					
				}
				MPI_Barrier(mpi2d.row.comm);
				t1 = MPI_Wtime() - t1;
				logInfo.nStoreImg += divVol->subTop.depth + divVol->subBottom.depth;
				logInfo.fStoreTime += t1 * 1000;
				
				size_t fSize = divVol->subTop.width*divVol->subTop.height*(divVol->subTop.depth + divVol->subBottom.depth) * 4;
				MyPrintf("WriteToFile, rank=%d, time=%f ms, WriteData=%f Gb, %f Gb/s\n", mpi2d.rank, t1 * 1000.f, (float)ToGB(fSize), (float)ToGB(fSize) / t1);
			}
		}		
	}
	{
		StopWatchWin watch;
		watch.start();
		threadFlt.reset();	
		threadBp.reset();	
		watch.stop();
		MyPrintf("Destroy:rank=%d, reset-time=%f, comm-time=%f\n", mpi2d.rank, watch.getAverageTime(), com.getAverageTime());
		this->logInfo.fThreadMain = com.getAverageTime();
	}

}
