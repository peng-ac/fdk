#ifndef __MPIWRAPER_H
#define __MPIWRAPER_H

#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include "../common/type.h" 
#include "../mnfdk/sharedMem.h"
#include "../common/IniFileFormat.h"
#include "../common/logger.h"
#include "../common/MyThread.h"
#include "../common/StopWatch.h"

enum DataStatus{
	Empty     = 0,
	Working   = 1,
	Ready     = 2,
	Done      = 3,
};

struct DataHeader{
	DataHeader(){
		memset(this, 0, sizeof(*this));
		strcpy(name, "dat");
		id = -1;
		width = height = 0;
		used_count = 0;
		header_size = sizeof(*this);
		total_size = header_size + sizeof(data[0])*width*height;
		bTranspose = true;
		bExit = false;
	}
	
	char name[4];
	int id;
	int status;
	int width;
	int height;
	int used_count;
	bool bTranspose;
	int header_size;
	int total_size;
	bool bExit;
	char data[0];
};

/////////////////////////////////////
struct MpiWraper;
struct MpiCompute;

struct MpiObj{
	MpiObj(){
	}
	virtual ~MpiObj(){
	}
	virtual void OnProc(){
	}
};

struct WorkerThread : public MyThread{
	WorkerThread(MpiObj* _pMpi);
	virtual~WorkerThread();
	virtual void OnProc();
	MpiObj* pMpi;
};

/////////////////////////////////////
struct Buffering {
	Buffering(const Parameter& para, int64 shmBlockCount, int64 _group, int64 _id_in_group);
public:
	std::vector<DataHeader*> vecShmRcv, vecShmSnd;
private:
	std::mutex shm_mtx;
	std::vector<char*> vecShmData;
	std::shared_ptr<SHM> pSHM;
	std::vector<char> buffer;
	int64 memBlockSize, memShmTotalSize, group, id_in_group;
};


/////////////////////////////////////
struct MpiWraper {
	struct CommandHeader {
		CommandHeader() {
			strcpy(name, "cmd");
			type = -1;
		}
		char name[4];
		int type;
		char data[128];
	};
	enum CommandType{
		SHM_READER = 10001,
		BP_READY_ONE   = 10003,
		BP_READY_ALL   = 10005,
	};
	enum MpiType {
		Computer = 0,
		Receiver = 1,
		Sender = 2,
	};
	MpiWraper(const Parameter& _para, int _group);
	int GetType() const;
	bool Run();
	int NextRank(int dir) const;
	virtual bool Rcv();
	virtual bool Snd();
	virtual bool Compute();
	virtual bool BpDone();
	//virtual bool Worker();
	int rank, size, nodes;
	int rank_new;
	int64 group, group_id, id_in_group;
	int64 shmBlockCount;
	std::shared_ptr<Buffering> pBuffering;
	const Parameter* pPara;

	std::shared_ptr<MpiCompute> pCompute;
};

struct MpiComputeRcv;
struct MpiComputeSnd;
struct MpiCompute : public MpiObj{
	MpiCompute(MpiWraper* _pMpi);
	virtual ~MpiCompute();
	virtual void OnProc();
	virtual void OnRcv();
	virtual void OnSnd();
	virtual void Compute();	
public:

	MpiWraper* pMpi;
	DataHeader* GetBuffer(SafeList<DataHeader*>& lst, int id=-1) const;
	SafeList<DataHeader*> list_free, list_filtered, list_computed;	
	BufferT<char> circle_buf;
	std::shared_ptr<MyThread> worker_thread;
	std::shared_ptr<MpiComputeRcv> rcv_thread;
	std::shared_ptr<MpiComputeSnd> snd_thread;
};

struct MpiComputeRcv : public MyThread {
	MpiComputeRcv(MpiCompute* _pCompute);
	virtual~MpiComputeRcv();
	virtual void OnProc();
	MpiCompute* pCompute;
};

struct MpiComputeSnd : public MyThread {
	MpiComputeSnd(MpiCompute* _pCompute);
	virtual~MpiComputeSnd();
	virtual void OnProc();
	MpiCompute* pCompute;
};

#endif // !__MPIWRAPER_H

