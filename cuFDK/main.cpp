#include "mpiFDK.h"

static int test(int argc, char **argv) 
{ 
	static const int N = 1;
	int i, myrank, nprocs; 
	int *send_buffer; 
	int *recv_buffer;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	send_buffer = new int[N]; for (i = 0; i < N; i++)
		send_buffer[i] = myrank;
	recv_buffer = new int[nprocs * N]; 
	MPI_Allgather(send_buffer, N, MPI_INT, recv_buffer, N, MPI_INT, MPI_COMM_WORLD); 
	for (i = 0; i < nprocs * N; i++) 
		fprintf(stderr, "myrank = %d, recv_buffer[%d] = %d\n", myrank, i, recv_buffer[i]); fprintf(stderr, "\n"); 
	delete[]recv_buffer; 
	delete[]send_buffer; 
	MPI_Finalize(); 
	return 0; 
}

static int test_mpi2d(int argc, char** argv)
{
//	DISPLAY_FUNCTION;
//	MPI_Init(&argc, &argv);
//	{
//		Mpi2D mpi2d;
//		int ranks_per_node = 2;
//		int world_rank, world_size;
//		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//		mpi2d.build(world_rank, world_size, world_size / ranks_per_node, ranks_per_node);
//	
//		printf("hostname=%s, world_rank=%d, world_size=%d, ranks_per_node=%d, mpi2d.row.size=%d, mpi2d.col.size=%d, x=%d, y=%d, pid=%d\n",
//			hostname().c_str(),
//			world_rank,
//			world_size,
//			ranks_per_node,
//			mpi2d.row.size,			
//			mpi2d.col.size,
//			mpi2d.row.rank,
//			mpi2d.col.rank,
//			getpid());
//		
//		
//		static const int N = 1;
//		int i, myrank, nprocs; 
//		int *send_buffer; 
//		int *recv_buffer;
//
//		myrank = mpi2d.col.rank;
//		nprocs = mpi2d.col.size;
//		send_buffer = new int[N]; 
//		for (i = 0; i < N; i++)
//			send_buffer[i] = myrank;
//	
//		recv_buffer = new int[nprocs * N]; 
//		MPI_Request request;
//		MPI_Status state;
//		MPI_Iallgather(send_buffer, N, MPI_INT, recv_buffer, N, MPI_INT, mpi2d.col.comm, &request); 
//		int ret = MPI_Wait(&request, &state);
//		for (i = 0; i < nprocs * N; i++) 
//			fprintf(stderr, "myrank = %d, recv_buffer[%d] = %d\n", myrank, i, recv_buffer[i]); fprintf(stderr, "\n"); 
//		delete[]recv_buffer; 
//		delete[]send_buffer; 
//	}
//	MPI_Finalize(); 
	return 0;
}

int test_mpi(int argc, char** argv)
{
	StopWatchWin watch;
	float tm[5];
	
	watch.start();
	// Initialize the MPI environment
	 MPI_Init(NULL, NULL);
	watch.stop();
	tm[0] = watch.getAverageTime();
	
	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	// Print off a hello world message
	printf("Hello world from processor %s, rank %d out of %d processors\n",
		processor_name,
		world_rank,
		world_size);

	watch.start();
	// Finalize the MPI environment.
	MPI_Finalize();	
	watch.stop();
	tm[1] = watch.getAverageTime();
	
	printf("MPI_Init time=%f, MPI_Finalize time=%f\n", tm[0], tm[1]);
}

int test_reduce(int argc, char** argv)
{
#if 1
	LogInfo logInfo;
	int ranks_per_node = 2;
	int rows = ranks_per_node;
	int volume_size = 512;	
	int nx(volume_size), ny(volume_size), nz(volume_size);
	int nu, nv;
	int projs;
	std::string src_dir, dst_dir;
	if (argc > 1) ranks_per_node = atoi(argv[1]);
	if (argc > 2) rows = atoi(argv[2]);
	if (argc > 3) nx = atoi(argv[3]);
	if (argc > 4) ny = atoi(argv[4]);
	if (argc > 5) nz = atoi(argv[5]);	
	if (argc > 6) nu = atoi(argv[6]);
	if (argc > 7) nv = atoi(argv[7]);
	if (argc > 8) projs = atoi(argv[8]);
	if (argc > 9) src_dir = argv[9];
	if (argc > 10) dst_dir = argv[10];
	
	Mpi2D mpi2d;
	
	int world_rank, world_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	{
		typedef float DataType ;
		size_t _buf_size = size_t(4) * 1024 * 1024 * 1024;
		BufferT<float> src, dst;
		try
		{
			src.malloc(_buf_size);
			dst.malloc(_buf_size);			
		}
		catch (...)
		{
			printf("malloc buffer error\n");
		}
		
		for (size_t i = 0; i < _buf_size; i++) {
			src.data()[i] = 1;
		}
		
		int cols = world_size / rows;
		printf("world_rank=%d, world_size=%d, rows=%d, cols=%d, ranks_per_node=%d\n", world_rank, world_size, rows, cols, ranks_per_node);
		mpi2d.build(world_rank, world_size, rows, cols, ranks_per_node);	
		
		size_t buf_size = 1024;
		size_t sz8g = buf_size*buf_size*buf_size*8;
		for (int k = 0; k < 1000; k++)
		{
			if (buf_size > _buf_size)
				break;
			if (buf_size == sz8g){ 
			//MPI_Barrier(MPI_COMM_WORLD);
			double t1 = 0;
			size_t repeat = 10;
			for (int s = 0; s < repeat+1; s++) {
				if (s == 1){
					MPI_Barrier(MPI_COMM_WORLD);
					t1 = MPI_Wtime();
				} 
					
				for (size_t t = 0; t < buf_size; t += INT_MAX) {
					auto pSnd = src.data() + t;
					auto pRcv = dst.data() + t;
					size_t count = MIN(INT_MAX, buf_size - t);
					MPI_Reduce(pSnd, pRcv, count, MPI_FLOAT, MPI_SUM, 0, mpi2d.col.comm);
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			t1 = MPI_Wtime() - t1;
			t1 /= repeat;
		
			if (mpi2d.col.rank == 0) {
				int sum = mpi2d.row.size*(mpi2d.col.size - 1) + mpi2d.row.rank;	
				sum = mpi2d.col.size;
				for (size_t i = 0; i < buf_size; i++) {
					if (dst.data()[i] != sum) {
						printf("****error, rank=%d, sum=%d\n", world_rank, dst.data()[i]);
						break;
					}
				}	

				//if (mpi2d.row.rank == 0)
				//	std::cout << StringFormat("data_size=%f GB, time=%f, speed=%f Gb/s\n", sz, t1, speed);
			}	
			double sz = double(buf_size) * 4 / 1024.0 / 1024.0 / 1024.0;
			double speed = sz / t1;
			std::cout << StringFormat("rank=%d, data_size=%f GB, time=%f, speed=%f Gb/s\n", world_rank, sz, t1, speed);
			}
			buf_size *= 2;
		}
	}
	MPI_Finalize();
#endif		
	return 0;
}

int main(int argc, char** argv)
{
//	std::string str;
//	for (int i = 0; i < argc; i++){
//		str += StringFormat("%s ", argv[i]);
//	}
//	printf("CMD : %s\n", str.c_str());
	
	extern int cuFDK(int argc, char **argv);
	
	//return test_mpi(argc, argv);
	//return test_mpi2d(argc, argv);
	//return test(argc, argv);
	//return test_reduce(argc, argv);
	return cuFDK(argc, argv);
}
