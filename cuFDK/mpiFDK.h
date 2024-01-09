#ifndef __MPIFDK_H
#define __MPIFDK_H

#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include "../common/type.h" 
//#include "../mnfdk/sharedMem.h"
#include "../common/IniFileFormat.h"
#include "../common/logger.h"
#include "../common/MyThread.h"
#include "../common/StopWatch.h"
#include "../cudaBP/backprojection.cuh"

#define MyPrintf printf
//#define MyPrintf

static const char* GetSaveHome(){
	static const char szHome[3][512] = { 
		"/groups2/gaa50004/gaa10008ku/",
		"/home/pengdadi/",
		"/gs/hs0/tgc-ebdcrest/chen/",
	};
	const char* home = szHome[0];
	if (IsPathExisted(szHome[0]))      home = szHome[0];
	else if (IsPathExisted(szHome[1])) home = szHome[1];
	return home;
}

struct TwoSubVolume{
	geo::BoxT<size_t> subTop, subBottom;
	float data[0];
};

struct SubVolume : public TwoSubVolume {
	SubVolume()
		: buffer(NULL)
		, buffer_size(0)
		, rank(0)
		, size(0) {
	}
	virtual~SubVolume() {
		if (buffer) delete[]buffer; buffer = NULL;
	}
	//geo::BoxT<int> subTop, subBottom;
	size_t nx, ny, nz;
	float* buffer;
	size_t buffer_size;
	size_t rank, size;
};

struct DivVolume : public SubVolume {
	DivVolume(size_t _nx, size_t _ny, size_t _nz, size_t _rank, size_t _size) {
		//DISPLAY_FUNCTION;
		VERIFY_TRUE(1<<int(std::log2(_size)) == _size);
		nx = _nx; ny = _ny; nz = _nz;
		rank = _rank; 
		size = _size;
		
		bool symmetry = true;
		if (nx >= 8192 && ny >= 8192) {
			symmetry = false;
		}
		
		size_t fx = 1;
		size_t fy = 1;
#if 1
		size_t fz = _size / fy / fx;
#else
		VERIFY_TRUE(_nz % (32 * 2) == 0);
		int fz = _nz / (32 * 2);
		if (nx >= 8192 && ny >= 8192){
			printf("For 8K volume\n");
			fz = _nz / (32);			
		}

		if (fz > _size){
			fz = _size;
		}
		fx = std::sqrt(_size / fz);
		fy = _size / fz / fx;
#endif
		
		size_t subNz = nz / fz;
		size_t subNy = ny / fy;
		size_t subNx = nx / fx;
		
		VERIFY_TRUE(fx == 1 && fy == 1);
		VERIFY_TRUE(fz*fy*fx == size);		
#if 0
		double subNx = std::sqrt(_size / fz);
		
		double len = std::cbrt(double(_nx)*_ny*_nz / _size);
		int subNz = int(len + 64 - 1 + 0.5) / 64 * 64;

		subNz = std::exp2(std::ceil(std::log2(subNz))) + 0.5; 
		//printf("subNz = %d\n", subNz);
		VERIFY_TRUE(_nz % subNz == 0);
		
		int64 subNx = subNz;		
		int64 subNy = double(_nx*_ny*_nz) / (_size*subNx*subNz) + 0.5;	
		while (subNy < subNz && subNy < _ny) {
			subNy *= 2; subNz /= 2;
		}
		//		int64 s = subNz / WarpSize;
		//		int64 xx = _nx / subNx;
		//		if (xx > s){		
		//		} else {
		//			
		//		}
		#endif			
		
		
			//printf("Nx,Ny,Nz=%d,%d,%d, cnt=%d\n", subNx, subNy, subNz, _size);
		
		if(symmetry == true){
			size_t sx = (_nx + subNx - 1) / subNx; 
			size_t sy = (_ny + subNy - 1) / subNy;
			size_t sz = (_nz + subNz - 1) / subNz;
			VERIFY_TRUE(sx*sy*sz <= _size);
		
			size_t z = _rank / (sx*sy);
			size_t y = (_rank - z*sx*sy) / sx;
			size_t x = _rank - z*sx*sy - y*sx;
		
			subTop = geo::BoxT<size_t>(x*subNx, y*subNy, z*subNz / 2, subNx, subNy, subNz / 2);
			subBottom = subTop;
			subBottom.z = _nz - 1 - subTop.z - subBottom.depth + 1;
			buffer_size = subTop.width*subTop.height*(subTop.depth + subBottom.depth);
			buffer = new float[buffer_size];
			//printf(StringFormat("divVol->buffer_size = : %f Gb\n", ToGB(buffer_size) * 4).c_str());
			printf("symmetry=true, rank=%d, size=%d, top: (%d,%d,%d_%dx%dx%d), bottom: (%d,%d,%d_%dx%dx%d), divVol->buffer_size=%f Gb\n",
				rank,
				size,
				subTop.x,
				subTop.y,
				subTop.z,
				subTop.width,
				subTop.height,
				subTop.depth,
				subBottom.x,
				subBottom.y,
				subBottom.z,
				subBottom.width,
				subBottom.height,
				subBottom.depth,
				ToGB(buffer_size) * 4);
			//std::cout << StringFormat("########################subTop.MinZ()=%d, subBottom.MaxZ()=%d, _nz =%d\n", subTop.MinZ(), subBottom.MaxZ(), _nz);
			VERIFY_TRUE(subTop.MinZ() + subBottom.MaxZ() == _nz - 1);
		}else{
			VERIFY_TRUE(nz %size == 0);
			subTop = geo::BoxT<size_t>(0, 0, nz / size*rank, nx, ny, nz / size);
			subBottom = subTop; subBottom.depth = 0;
			buffer_size = subTop.width*subTop.height*(subTop.depth + subBottom.depth);
			buffer = new float[buffer_size];
			//printf(StringFormat("divVol->buffer_size = : %f Gb\n", ToGB(buffer_size) * 4).c_str());
			printf("symmetry=false, rank=%d, size=%d, top: (%d,%d,%d_%dx%dx%d), bottom: (%d,%d,%d_%dx%dx%d), divVol->buffer_size=%f Gb\n",
				rank,
				size,
				subTop.x,
				subTop.y,
				subTop.z,
				subTop.width,
				subTop.height,
				subTop.depth,
				subBottom.x,
				subBottom.y,
				subBottom.z,
				subBottom.width,
				subBottom.height,
				subBottom.depth,
				ToGB(buffer_size) * 4);
			//std::cout << StringFormat("########################subTop.MinZ()=%d, subBottom.MaxZ()=%d, _nz =%d\n", subTop.MinZ(), subBottom.MaxZ(), _nz);
			VERIFY_TRUE(subBottom.depth == 0);			
		}
	}
};	

struct DataHeader {
	short id;
	char name[16];
	short bTranspose;
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

struct Info {
	Info(){
		memset(this, 0, sizeof(*this));
	}
	int blk_per_rank;
	int blk_per_comm;
	int64 imgSize, blkSize, totalSize;
	int start_idx, end_idx;
};

struct LogInfo{
	LogInfo() {
		memset(this, 0, sizeof(*this));
	}
	void Show(const Parameter& para){
		float scale = 0.001;
		double gup = double(para.nx * 0.001)*double(para.nx * 0.001)*double(para.nz * 0.001)*para.nProj / double(fTotalTime / 1000.0);
		{
			std::string str;
			str += StringFormat("fMpiInitTime=%f, fFinalizeTime=%f\n", fMpiInitTime*scale, fFinalizeTime*scale);
			str += StringFormat("rank=%d, size=%d, rows=%d, cols=%d, x=%d, y=%d\n", rank, size, rows, cols, x, y);
			str += StringFormat("cpuCores=%d, fFilterInitTime=%f, fCudaInitTime=%f\n", cpuCores, fFilterInitTime*scale, fCudaInitTime*scale);
			str += StringFormat("fAllGatherTime=%f, fReduceTime=%f\n", fAllGatherTime*scale, fReduceTime*scale);
			str += StringFormat("fLoadFltTime=%f, fBpTime=%f, fStoreTime=%f, fTotalTime=%f\n",
				fLoadFltTime*scale,
				fBpTime*scale,
				fStoreTime*scale,
				fTotalTime*scale);
			std::cout << str;
		}
		{
			std::string str;
			str += StringFormat("%05d-%05d-%05d %05d-%05d-%05d, %05d, %05d, %05d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
				para.nu,
				para.nv,
				para.nProj,
				para.nx,
				para.ny,
				para.nz, 
				size,
				cols,
				rows,
				fMpiInitTime*scale,
				fFilterInitTime*scale,
				fCudaInitTime*scale,
				fLoadFltTime*scale,
				fBpTime*scale,
				fD2HTime*scale,
				fAllGatherTime*scale,
				fReduceTime*scale,
				fStoreTime*scale,
				fFinalizeTime*scale,
				fThreadFlt*scale,
				fThreadMain*scale,
				fThreadBp*scale,
				fTotalTime*scale,
				gup);
			std::string name = StringFormat("%05d-%05d-%05d.%05d-%05d-%05d.%06d.t2",
				para.nu,
				para.nv,
				para.nProj,
				para.nx,
				para.ny,
				para.nz, 
				size);
			FILE* fp = fopen(name.c_str(), "a");
			if (fp){
				fprintf(fp, str.c_str());
				fclose(fp);
			} else {
				printf("write log error\n");
			}
		}
		{		
			std::string str;
			str += StringFormat("%05d-%05d-%05d->%05d-%05d-%05d, %05d, %.3f, %f\n",
				para.nu,
				para.nv,
				para.nProj,
				para.nx,
				para.ny,
				para.nz, 
				size,
				fTotalTime,
				(float)gup
				);
			std::string name = StringFormat("%05d-%05d-%05d_%05d-%05d-%05d.%06d.t1",
				para.nu,
				para.nv,
				para.nProj,
				para.nx,
				para.ny,
				para.nz, 
				size);
			FILE* fp = fopen(name.c_str(), "w");
			if (fp) {
				fprintf(fp, str.c_str());
				fclose(fp);
			}
			else {
				printf("write log error\n");
			}
		}
	}
	int cpuCores;
	int rank, size, rows, cols, x, y;
	int nLoadImg, nStoreImg;
	float fStoreTime;
	float fFilterInitTime, fCudaInitTime;
	float fMpiInitTime, fFinalizeTime;
	float fLoadFltTime, fBpTime, fD2HTime;
	float fAllGatherTime, fReduceTime, fTotalTime;
	float fThreadFlt, fThreadMain, fThreadBp;
	float rsv[32];
};

struct mpiFDK;

struct FilterThread : public MyThread{
	FilterThread(mpiFDK* _pMPI);
	virtual void OnProc();
	mpiFDK* pMPI;
	SafeList<DataHeader*> lstEmpty, lstReady;
public:
	//int blk_per_rank, blk_per_comm;
	//int64 imgSize, blkSize, totalSize;
private:
	BufferT<char> buffer;
};

struct BpThread : public MyThread {
	BpThread(mpiFDK* _pMPI);
	virtual ~BpThread();
	void OnProc();
private:
	mpiFDK* pMPI;
	void PushBpData(int id, const float* pImg, const float* pMat);
	std::shared_ptr<BackProjectionFDK> pBP;
	BufferT<float> projImg;
	BufferT<float> projMat;
	size_t nNum;
	std::string strMsg;
};

struct Mpi1D{
	Mpi1D() {
	}
	virtual ~Mpi1D(){
		//MPI_Comm_free(&comm);
	}
	void build(int _rank, int _size, int _color, MPI_Comm world = MPI_COMM_WORLD) {
		color = _color;
		MPI_Comm_split(world, color, _rank, &comm);
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
	}
	int color, rank, size;
	MPI_Comm comm;
};

struct Mpi2D{
	Mpi2D(){
	}
	void build(int _rank, int _size, int _rows, int _cols, int _ranks_per_node) {
		rank = _rank; size = _size;
		col.build(_rank, _size, _rank % _cols, MPI_COMM_WORLD);
		row.build(_rank, _size, _rank / _cols, MPI_COMM_WORLD);	
		//row_col.build(row.rank, row.size, row.rank % _ranks_per_node, row.comm);
		//row_row.build(row.rank, row.size, row.rank / _ranks_per_node, row.comm);
		//printf("row.rank=%d, row.size=%d, _ranks_per_node =%d, row_row.size = %d, row_col.size = %d\n", row.rank, row.size, _ranks_per_node, row_row.size, row_col.size);
		//VERIFY_TRUE(row_row.size == _ranks_per_node);
		//VERIFY_TRUE(row.size == row_row.size*row_col.size);
	}
	Mpi1D row, col;
	int rank, size;
};

struct mpiFDK{
	mpiFDK(const Parameter& _para, LogInfo& _logInfo, int _ranks_per_node = 4, int rows = 4);
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
	Info& initInfo();
	bool bThreadFltDone, bAllGatherDone, bBpDone;
	Info info;
	std::shared_ptr<DivVolume> divVol;
	LogInfo& logInfo;
private:
	BufferT<char> buffer;
};

#endif //__MPIFDK_H
