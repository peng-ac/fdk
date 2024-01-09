#ifndef __MPIFDK_H
#define __MPIFDK_H

#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include "../common/type.h" 
#include "../mnfdk/sharedMem.h"
#include "../common/IniFileFormat.h"
#include "../common/logger.h"
#include "../common/MyThread.h"
#include "../common/StopWatch.h"

struct DataHeader {
	int id;
	float projMat[12];
	int head_size;
	int total_size;
	int dim[2];
	char data[0];
};

struct MultiDataHeader{
	int data_count;
	int data_size;
	char data[0];
};

struct mpiFDK;

struct FilterThread : public MyThread{
	enum{
		MAX_IMG_COUNT = 32,
	};
	FilterThread(mpiFDK* _pMPI);
	virtual void OnProc();
	mpiFDK* pMPI;
	SafeList<DataHeader*> lstEmpty, lstReady;
	bool bDone;
public:
	int blk_per_rank, blk_per_comm;
	int64 imgSize, blkSize, totalSize;
private:
	BufferT<char> buffer;
};

struct BpThread : public MyThread {
	BpThread(mpiFDK* _pMPI);
	virtual ~BpThread();
	void OnProc();
private:
	mpiFDK* pMPI;
};

struct Mpi1D{
	Mpi1D() {
	}
	virtual ~Mpi1D(){
		//MPI_Comm_free(&comm);
	}
	void build(int _rank, int _size, int _color){
		color = _color;
		MPI_Comm_split(MPI_COMM_WORLD, color, _rank, &comm);
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
	}
	int color, rank, size;
	MPI_Comm comm;
};

struct Mpi2D{
	Mpi2D(){
	}
	void build(int _rank, int _size, int _rows, int _cols){
		rank = _rank; size = _size;
		col.build(_rank, _size, _rank % _cols);
		row.build(_rank, _size, _rank / _cols);
	}
	Mpi1D row, col;
	int rank, size;
};

struct mpiFDK{
	mpiFDK(const Parameter& _para, int _ranks_per_node = 2);
	bool Run();
	virtual ~mpiFDK();
public:
	std::shared_ptr<FilterThread> threadFlt;
	std::shared_ptr<BpThread> threadBp;
	const Parameter& para;
	int world_rank, world_size, ranks_per_node;
	Mpi2D mpi2d;
	
	SafeList<char*> lstEmpty, lstReady;
	int max_multi_data_count;
	Point2dT<int> GetXY() const;
private:
	BufferT<char> buffer;
};

#endif //__MPIFDK_H