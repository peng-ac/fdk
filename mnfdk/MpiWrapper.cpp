#include "MpiWrapper.h"
#include "../Filterlib/FilterLib.h"

#define ABORT_ICHIGO 15

/////////////////////////////////////////////////////////////////////////////////////
WorkerThread::WorkerThread(MpiObj* _pMpi) {
	pMpi = _pMpi;
}
WorkerThread::~WorkerThread() {
}
void WorkerThread::OnProc() {
	VERIFY_TRUE(pMpi);
	std::cout << "Begin WorkerThread" << std::endl;
	pMpi->OnProc();
	std::cout << "End   WorkerThread" << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////


Buffering::Buffering(const Parameter& para, int64 shmBlockCount, int64 _group, int64 _id_in_group) {
	group = _group;
	id_in_group = _id_in_group;
	SHM::SHM_TYPE type = id_in_group == 0 ? SHM::SERVER : SHM::CLIENT;
	memBlockSize = sizeof(DataHeader) + para.nu*para.nv * sizeof(float);
	memShmTotalSize = memBlockSize * shmBlockCount*(group - 1);
	VERIFY_TRUE(group % 2 == 1);
	pSHM = std::make_shared<SHM>(type, memShmTotalSize);
	assert(pSHM);
	assert(pSHM->buffer);

	char* pRcv = pSHM->buffer;
	char* pSnd = pSHM->buffer + memBlockSize * shmBlockCount*(group - 1)/2;
	if (type == SHM::SERVER) { 
		DataHeader tmp;
		tmp.width = para.nv;
		tmp.height = para.nu;
		tmp.total_size = tmp.header_size + sizeof(float)*para.nu*para.nv;
		tmp.status = DataStatus::Empty;
		memset(pSHM->buffer, 0, memShmTotalSize);
		vecShmRcv.resize(shmBlockCount*(group - 1) / 2);
		vecShmSnd.resize(vecShmRcv.size());
		for (int64 i = 0; i < vecShmRcv.size(); i++) {
			vecShmRcv[i] = (DataHeader*)pRcv;
			memcpy(vecShmRcv[i], &tmp, sizeof(tmp));
			pRcv += memBlockSize;
		}
		for (int64 i = 0; i < vecShmSnd.size(); i++) {
			vecShmSnd[i] = (DataHeader*)pSnd;
			memcpy(vecShmSnd[i], &tmp, sizeof(tmp));
			pSnd += memBlockSize;
		}		
		VERIFY_TRUE(pRcv == pSHM->buffer + memBlockSize * shmBlockCount*(group - 1) / 2);
		VERIFY_TRUE(pSnd == pSHM->buffer + memBlockSize * shmBlockCount*(group - 1));
	} else if (_id_in_group <= _group / 2){
		vecShmRcv.resize(shmBlockCount);
		pRcv += (_id_in_group-1)*memBlockSize;
		for (int64 i = 0; i < vecShmRcv.size(); i++){
			vecShmRcv[i] = (DataHeader*)pRcv;
			pRcv += memBlockSize*(group - 1) / 2;
		}
	} else if (_id_in_group > _group / 2){
		vecShmSnd.resize(shmBlockCount);
		int64 id = _id_in_group - _group / 2 - 1;
		pSnd += id*memBlockSize;
		for (int64 i = 0; i < vecShmSnd.size(); i++) {
			vecShmSnd[i] = (DataHeader*)pSnd;
			pSnd += memBlockSize*(group - 1) / 2;
		}		
	}
}

///////////////////////////////////////////////////////////////////////////////
MpiWraper::MpiWraper(const Parameter& _para, int _group) {
	//comm = MPI_COMM_WORLD;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	printf("hostname=%s, rank=%d, size=%d, pid=%d\n", hostname().c_str(), rank, size, getpid());
#if DEBUG==1
	const char* dbgfile = "__dbg.txt";
	DeleteFile(dbgfile);
	while (!IsPathExisted(dbgfile)) sleep(1);
	std::cout << "dbg.txt ok!" << std::endl;
#endif

	//Init
	group = _group;
	group_id = rank / group;
	id_in_group = rank % group;
	nodes = size / group;
	VERIFY_TRUE(size % group == 0);

	shmBlockCount = 3;
	pPara = &_para;
}
int MpiWraper::GetType() const {
	if (id_in_group == 0)              return MpiType::Computer;
	else if (id_in_group <= group / 2) return MpiType::Receiver;
	else                               return MpiType::Sender;
}
bool MpiWraper::Run() {
	int type = GetType();
	switch (type) {
	case MpiType::Computer:
		return Compute();
		break;
	case MpiType::Receiver:
		return Rcv();
		break;
	case MpiType::Sender:
		return Snd();
		break;
	default:
		assert(0);
		break;
	}
	return true;
}
int MpiWraper::NextRank(int dir) const {
	int id = rank + group * dir;
	while (id >= size) id -= size;
	while (id < 0) id += size;
	return id;
}
	
bool MpiWraper::Rcv() {
	const int id = NextRank(-1) + group / 2;
	printf("%s,rank=%d,size=%d,group=%d,group_id=%d,id_in_group=%d\n", hostname().c_str(), rank, size, group, group_id, id_in_group);
	printf("%s : %d Rcv from : %d\n", hostname().c_str(), rank, id);
	MPI_Status state;
	{
		//Init
		CommandHeader header;
		int rev = (rank / group)*group;
		MPI_Recv(&header, sizeof(header), MPI_CHAR, rev, 0, MPI_COMM_WORLD, &state);
		printf("%s:Rcv%d got SHMREADY from: %d\n", hostname().c_str(), rank, rev);
		VERIFY_TRUE(header.type == SHM_READER);
		pBuffering = std::make_shared<Buffering>(*pPara, shmBlockCount, group, id_in_group);
	}
	{	
		std::vector<DataHeader*>& vecShmRcv = pBuffering->vecShmRcv;
		VERIFY_TRUE(!vecShmRcv.empty());
		
		for (;;){
			DataHeader* pHeader = NULL;
			for (int i = 0; i < vecShmRcv.size(); i++) {
				if (vecShmRcv[i]->status == DataStatus::Empty){
					pHeader = vecShmRcv[i];
					break;
				}			
			}
			if (!pHeader){
				std::this_thread::sleep_for(std::chrono::microseconds(1));
				continue;
			}
			
			VERIFY_TRUE(pHeader->status == DataStatus::Empty);
			MPI_Request request;
			//printf("#%d Rcv to : %d#....%s,rank=%d,size=%d,group=%d,group_id=%d,id_in_group=%d,total_size=%d\n", rank, id, hostname().c_str(), rank, size, group, group_id, id_in_group, pHeader->total_size);	
			int ret = MPI_Irecv(pHeader, pHeader->total_size, MPI_CHAR, id, 0, MPI_COMM_WORLD, &request);
			if (ret != MPI_SUCCESS) {
				std::cout << "error:MPI_Irecv" << ret << std::endl;
				MPI_Abort(MPI_COMM_WORLD, ABORT_ICHIGO);
			}			
			ret = MPI_Wait(&request, &state);
			if (ret != MPI_SUCCESS) {
				std::cout << "error:MPI_Wait" << ret << std::endl;
				MPI_Abort(MPI_COMM_WORLD, ABORT_ICHIGO);
			}
			VERIFY_TRUE(pHeader->status == DataStatus::Working);
			pHeader->status = DataStatus::Ready;
			//printf("rcv--------------------------------->ok\n");
			if(pHeader->bExit){
				printf("~~~Exit: rank=%d\n", rank);
				break;
			}
		}	
	}
	return true;
}

bool MpiWraper::Snd() {
	const int id = NextRank(1) - group / 2;
	printf("%s,rank=%d,size=%d,group=%d,group_id=%d,id_in_group=%d\n", hostname().c_str(), rank, size, group, group_id, id_in_group);
	printf("%d Snd to : %d\n", rank, id);
	MPI_Status state;
	{
		//Init
		CommandHeader header;
		int rev = (rank / group)*group;
		MPI_Recv(&header, sizeof(header), MPI_CHAR, rev, 0, MPI_COMM_WORLD, &state);
		printf("%s:Rcv%d got SHMREADY from: %d\n", hostname().c_str(), rank, rev);
		VERIFY_TRUE(header.type == SHM_READER);
		pBuffering = std::make_shared<Buffering>(*pPara, shmBlockCount, group, id_in_group);
	}
	{
		std::vector<DataHeader*>& vecShmSnd = pBuffering->vecShmSnd;
		VERIFY_TRUE(!vecShmSnd.empty());

		for (;;) {
			DataHeader* pHeader = NULL;
			for (int i = 0; i < vecShmSnd.size(); i++) {
				if (vecShmSnd[i]->status == DataStatus::Ready) {
					pHeader = vecShmSnd[i];
					break;
				}			
			}
			if (!pHeader) {
				std::this_thread::sleep_for(std::chrono::microseconds(1));
				continue;
			}		
		
			VERIFY_TRUE(pHeader->status == DataStatus::Ready);
			if (pHeader->bExit) {
				VERIFY_TRUE(pHeader->id == -1);
			}else {
				VERIFY_TRUE(pHeader->id >= 0 && pHeader->id < pPara->nProj); 
				VERIFY_TRUE(pHeader->used_count > 0 && pHeader->used_count < pPara->nProj);
			}		
			pHeader->status = DataStatus::Working;
			MPI_Request request;
			//printf("#%d Snd to : %d#....%s,rank=%d,size=%d,group=%d,group_id=%d,id_in_group=%d, total_size=%d\n", rank, id, hostname().c_str(), rank, size, group, group_id, id_in_group, pHeader->total_size);
			int ret = MPI_Isend(pHeader, pHeader->total_size, MPI_CHAR, id, 0, MPI_COMM_WORLD, &request);
			if (ret != MPI_SUCCESS) {
				std::cout << "error:MPI_Isend" << ret << std::endl;
				MPI_Abort(MPI_COMM_WORLD, ABORT_ICHIGO);
			}
			ret = MPI_Wait(&request, &state);
			if (ret != MPI_SUCCESS) {
				std::cout << "error:MPI_Wait" << ret << std::endl;
				MPI_Abort(MPI_COMM_WORLD, ABORT_ICHIGO);
			}
			if (!pHeader->bExit) pHeader->status = DataStatus::Empty;
			if (pHeader->bExit) {
				printf("~~~Exit: rank=%d\n", rank);
				break;
			}
		}		
	}
	return true;
}

bool MpiWraper::Compute() {
	printf("%s,rank=%d,size=%d,group=%d,group_id=%d,id_in_group=%d\n", hostname().c_str(), rank, size, group, group_id, id_in_group);
	printf("%d compute : %d\n", rank, id_in_group);
	{
		system(StringFormat("sh ./clearSMEM.sh").c_str());
		pBuffering = std::make_shared<Buffering>(*pPara, shmBlockCount, group, id_in_group);
		pCompute = std::make_shared<MpiCompute>(this);	
		{
			CommandHeader header;
			header.type = CommandType::SHM_READER;
			for (int i = 1; i < group; i++)
				MPI_Send(&header, sizeof(header), MPI_CHAR, i + rank, 0, MPI_COMM_WORLD);
		}
		pCompute->Compute();
		pCompute.reset();			
	}
	return true;
}

bool MpiWraper::BpDone()
{
	if (id_in_group == 0) {
		MpiWraper::CommandHeader header;			
		MPI_Status state;
		if (rank == 0) {
			for (int i = 1; i < nodes; i++) {
				MPI_Recv(&header, sizeof(header), MPI_CHAR, i*group, 0, MPI_COMM_WORLD, &state);
				VERIFY_TRUE(header.type == MpiWraper::CommandType::BP_READY_ONE);
			}				
			header.type = MpiWraper::CommandType::BP_READY_ALL;
			for (int i = 1; i < nodes; i++) 
				MPI_Send(&header, sizeof(header), MPI_CHAR, i*group, 0, MPI_COMM_WORLD);
		}
		else {
			header.type = MpiWraper::CommandType::BP_READY_ONE;
			MPI_Send(&header, sizeof(header), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
			MPI_Recv(&header, sizeof(header), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &state);
			VERIFY_TRUE(header.type == MpiWraper::CommandType::BP_READY_ALL);
		}
	}
	std::cout << "=====>Done, rank=" << rank <<std::endl;
	return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
MpiCompute::MpiCompute(MpiWraper* _pMpi) : pMpi (_pMpi) 
{
	worker_thread = std::make_shared<WorkerThread>(this);
	worker_thread->Create();
		
	rcv_thread = std::make_shared<MpiComputeRcv>(this); 
	snd_thread = std::make_shared<MpiComputeSnd>(this); 
	
	rcv_thread->Create();
	snd_thread->Create();
}

MpiCompute::~MpiCompute()
{
	rcv_thread.reset();
	snd_thread.reset();
	worker_thread.reset();
}

DataHeader* MpiCompute::GetBuffer(SafeList<DataHeader*>& lst, int id = -1) const
{
	//DISPLAY_FUNCTION;
	//std::cout << "Begin get free buffer" << std::endl;
	DataHeader* pHeader = NULL;
	while (1) {
		lst.get_front(&pHeader);
		if (pHeader) break;
		if (id >= 0) printf("GetBuffer id=%d\n", id);
		std::this_thread::sleep_for(std::chrono::microseconds(1));
	}
	VERIFY_TRUE(pHeader);
	return pHeader;
	//std::cout << "End get free buffer" << std::endl;
}

//in main thread
void MpiCompute::Compute()
{
	const Parameter& para = *pMpi->pPara;
	const int MAX_PROJ_COUNT = 32;
	BufferT<float> projImgs, projMats;	
	const int PROJ_IMG_SIZE = para.nu*para.nv;
	const int PROJ_MAT_SIZE = 12;
	projImgs.malloc(PROJ_IMG_SIZE*MAX_PROJ_COUNT);
	projMats.malloc(PROJ_MAT_SIZE*MAX_PROJ_COUNT);
	VERIFY_TRUE(projImgs.data() && projMats.data());
	int index = 0;
	int total_num = 0;
	while (1) {
		DataHeader* pHeader = NULL;
		if (list_filtered.size() < 0) {
			std::this_thread::sleep_for(std::chrono::microseconds(1));
			continue;		
		}
		pHeader = GetBuffer(list_filtered); 
		VERIFY_TRUE(pHeader);
		memcpy(&projImgs[PROJ_IMG_SIZE*index], pHeader->data, PROJ_IMG_SIZE*sizeof(float));
		memcpy(&projMats[PROJ_MAT_SIZE*index], &para.vec_rot_mat_ijk3x4[PROJ_MAT_SIZE*pHeader->id], PROJ_MAT_SIZE*sizeof(float));
		index++;
		total_num++;
		if (index >= PROJ_MAT_SIZE){			
			index = 0;
		}			
		VERIFY_TRUE(pHeader->width*pHeader->height == para.nu*para.nv);
		VERIFY_TRUE(pHeader->total_size == para.nu*para.nv*sizeof(float)+sizeof(*pHeader));
		pHeader->used_count++;
		if (pHeader->used_count >= this->pMpi->nodes){
			//printf("+++++++++rank=%d,id=%d, used_count=%d, nodes=%d, w=%d, h=%d, Header=0x%x\n", pMpi->rank, pHeader->id, pHeader->used_count, pMpi->nodes, pHeader->width, pHeader->height, pHeader);
			//WriteToFile<char, char>(StringFormat("%sflt/%04d.raw", para.str_working_folder.c_str(), pHeader->id), pHeader->data, pHeader->total_size - pHeader->header_size);
			list_free.push_back(pHeader);
		}else{
			//printf("---------rank=%d,id=%d, used_count=%d, nodes=%d, w=%d, h=%d, Header=0x%x\n", pMpi->rank, pHeader->id, pHeader->used_count, pMpi->nodes, pHeader->width, pHeader->height, pHeader);
			list_computed.push_back(pHeader);
		}			
		if (total_num >= para.nProj){
			printf("-->rank=%d,proc_count=%d\n", pMpi->rank, total_num);
			break;
		}
	}
	VERIFY_TRUE(pMpi->BpDone() == true);
	{		
		snd_thread->m_bExit = true;
		rcv_thread->m_bExit = true;
		worker_thread->m_bExit = true;
		worker_thread->Destroy();
		printf("worker_thread->Destroy(), rank=%d\n", pMpi->rank);

		std::vector<DataHeader*>& vecShmSnd = pMpi->pBuffering->vecShmSnd;
		for (int i = 0; i < vecShmSnd.size(); i++) {
			vecShmSnd[i]->bExit = true;
			vecShmSnd[i]->id = -1;
			vecShmSnd[i]->status = DataStatus::Ready;
		}			
		
		snd_thread->Destroy();	 printf("snd_thread->Destroy(), rank=%d\n", pMpi->rank);	
		rcv_thread->Destroy();	 printf("rcv_thread->Destroy(), rank=%d\n", pMpi->rank);

		printf("^^^^ all exit\n");
	}	
}

void MpiCompute::OnRcv()
{
	printf("Begin MpiCompute::OnRcv(), rank=%d\n", pMpi->rank);
	bool& bExit = rcv_thread->m_bExit;
	while (!bExit) {
		DataHeader* pHeader = NULL;
		if (list_free.size() == 0){
			std::this_thread::sleep_for(std::chrono::microseconds(1));
			continue;		
		}
		pHeader = GetBuffer(list_free, pMpi->rank); 		
		VERIFY_TRUE(pHeader);
		VERIFY_TRUE(pMpi->pBuffering);
		std::vector<DataHeader*>& vecShmRcv = pMpi->pBuffering->vecShmRcv;
		while (!bExit) {
			DataHeader* p = NULL;
			for (int i = 0; i < vecShmRcv.size() && !bExit; i++) {
				if (vecShmRcv[i]->status == DataStatus::Ready){
					p = vecShmRcv[i];
					break;
				}
			}
			if (bExit)
				break;
			if (p){
				if (p->bExit){
					VERIFY_TRUE(p->id == -1);
				}else{
					VERIFY_TRUE(p->id >= 0 && p->id < pMpi->pPara->nProj);
					VERIFY_TRUE(p->used_count > 0 && p->used_count < pMpi->pPara->nProj);					
				}
				memcpy(pHeader, p, p->total_size);
				p->id = -1;
				p->status = DataStatus::Empty;
				pHeader->status = DataStatus::Working;
				if (pHeader->bExit) list_free.push_back(pHeader);
				else                list_filtered.push_back(pHeader);
				pHeader = NULL;
				break;
			}
		}
		if (pHeader){
			//for safe exit
			list_free.push_back(pHeader);
		}
	}
	//printf("End MpiCompute::OnRcv(), rank=%d\n", pMpi->rank);
}

void MpiCompute::OnSnd()
{
	bool& bExit = snd_thread->m_bExit;
	while (!bExit) {
		DataHeader* pHeader = NULL;
		if (list_computed.size() == 0) {
			std::this_thread::sleep_for(std::chrono::microseconds(1));
			continue;		
		}
		pHeader = GetBuffer(list_computed); 		
		VERIFY_TRUE(pHeader);
		VERIFY_TRUE(pMpi->pBuffering);
		std::vector<DataHeader*>& vecShmSnd = pMpi->pBuffering->vecShmSnd;
		
		while (!bExit) {
			DataHeader* p = NULL;
			for (int i = 0; i < vecShmSnd.size() && !bExit; i++) {
				if (vecShmSnd[i]->status == DataStatus::Empty) {
					p = vecShmSnd[i];
					break;
				}
			}
			if (bExit)
				break;
			if (p) {		
				VERIFY_TRUE(pHeader->id >= 0 && pHeader->id < pMpi->pPara->nProj);
				VERIFY_TRUE(pHeader->used_count > 0 && pHeader->used_count < pMpi->pPara->nProj);
				pHeader->status = DataStatus::Working;
				memcpy(p, pHeader, p->total_size);
				p->status = DataStatus::Ready;
				pHeader->status = DataStatus::Working;
				pHeader->id = -1;
				list_free.push_back(pHeader);
				pHeader = NULL;
				p = NULL;
				break;
			}	
		}
		if (pHeader) {
			//for safe exit
			list_filtered.push_back(pHeader);
		}
	}
}

void MpiCompute::OnProc()
{
	DISPLAY_FUNCTION;
	//it is in a single thread
	std::cout << "Begin Worker..." << std::endl;
	VERIFY_TRUE(pMpi);
	VERIFY_TRUE(pMpi->GetType() == MpiWraper::MpiType::Computer);
	const Parameter& para = *pMpi->pPara;
	{
		const int64 COUNT = 64;
		const int64 blocksize = sizeof(float)*para.nu*para.nv*COUNT + sizeof(DataHeader);
		const int64 sz = blocksize*COUNT;
		VERIFY_TRUE(circle_buf.malloc(sz));
		char* p = circle_buf.data();
		for (int64 i = 0; i < COUNT; i++) {
			list_free.push_back((DataHeader*)p);
			p += blocksize;
		}
		std::cout << "init list_free size = " << list_free.size() << std::endl;
	}
	{
		const int64 step = (para.nProj + pMpi->nodes - 1) / pMpi->nodes;
		int64 start = (pMpi->rank / pMpi->group)*step;
		int64 end = std::min(start + step, para.nProj);
		printf("size=%d, group=%d, start=%d, end=%d, step=%d\n", pMpi->size, pMpi->group, start, end, step);
		printf("start=%d,end=%d\n", start, end);
		const int order = log(para.fft_kernel_real.size()) / log(2) + 0.5;
		FilterImageRowMulti<2> FilterImageRow(order);
		for (int i = start; i < end; i++) {
			VERIFY_TRUE(!worker_thread->m_bExit);
			char szPath[1024];
			sprintf(szPath, "%simg%04d.raw", para.str_proj_img_folder.c_str(), i);
			VERIFY_TRUE(FileSize(szPath) == sizeof(float)*para.nu*para.nv);
			{
				ImageT<float> proj(para.nu, para.nv);
				VERIFY_TRUE(LoadFromFile(szPath, proj.buffer, proj.width*proj.height));
				auto src = proj.GetFrame();
				DataHeader* pHeader = 0;
				pHeader = GetBuffer(list_free);
				VERIFY_TRUE(pHeader);
				//WriteToFile<float, float>(StringFormat("%simg%04d.raw", para.str_working_folder.c_str(), i), proj.buffer, proj.width*proj.height);					
				bool bTranspose = true;
				pHeader->used_count = 0;
				if (bTranspose){
					pHeader->width = para.nv;
					pHeader->height = para.nu;	
				}else{
					pHeader->width = para.nu;
					pHeader->height = para.nv;					
				}
				pHeader->header_size = sizeof(*pHeader);
				pHeader->total_size  = pHeader->header_size + sizeof(float)*pHeader->width*pHeader->height;
				pHeader->bTranspose = bTranspose;
				pHeader->id = i;
				pHeader->status = DataStatus::Working;	
				FrameT<float> dst((float*)pHeader->data, pHeader->height, pHeader->width, i);				

				FilterEngine flt(para.cos_image_table.cos_sigma.GetFrame(), para.fft_kernel_real_f32);
				flt.Filtering(src, dst, bTranspose);
				
				//WriteToFile<float, float>(StringFormat("%s/img%04d.raw", para.str_working_folder.c_str(), i), dst.buffer, dst.width*dst.height);			
				VERIFY_TRUE(pHeader->id >= start && pHeader->id < end);
				list_filtered.push_back(pHeader);
			}
		}
	}
	std::cout << "End   Worker..." << std::endl;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

MpiComputeRcv::MpiComputeRcv(MpiCompute* _pCompute) : pCompute (_pCompute) 
{
	VERIFY_TRUE(pCompute);
}
MpiComputeRcv::~MpiComputeRcv() {
}
void MpiComputeRcv::OnProc() {
	DISPLAY_FUNCTION;
	pCompute->OnRcv();
}



MpiComputeSnd::MpiComputeSnd(MpiCompute* _pCompute)
	: pCompute (_pCompute) {
	VERIFY_TRUE(pCompute);
}
MpiComputeSnd::~MpiComputeSnd() {
}
void MpiComputeSnd::OnProc() {
	DISPLAY_FUNCTION;
	pCompute->OnSnd();
}
