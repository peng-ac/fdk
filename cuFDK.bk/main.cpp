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
	DISPLAY_FUNCTION;
	MPI_Init(&argc, &argv);
	{
		Mpi2D mpi2d;
		int ranks_per_node = 2;
		int world_rank, world_size;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		mpi2d.build(world_rank, world_size, world_size / ranks_per_node, ranks_per_node);
	
		printf("hostname=%s, world_rank=%d, world_size=%d, ranks_per_node=%d, mpi2d.row.size=%d, mpi2d.col.size=%d, x=%d, y=%d, pid=%d\n",
			hostname().c_str(),
			world_rank,
			world_size,
			ranks_per_node,
			mpi2d.row.size,			
			mpi2d.col.size,
			mpi2d.row.rank,
			mpi2d.col.rank,
			getpid());
		
		
		static const int N = 1;
		int i, myrank, nprocs; 
		int *send_buffer; 
		int *recv_buffer;

		myrank = mpi2d.col.rank;
		nprocs = mpi2d.col.size;
		send_buffer = new int[N]; 
		for (i = 0; i < N; i++)
			send_buffer[i] = myrank;
	
		recv_buffer = new int[nprocs * N]; 
		MPI_Request request;
		MPI_Status state;
		MPI_Iallgather(send_buffer, N, MPI_INT, recv_buffer, N, MPI_INT, mpi2d.col.comm, &request); 
		int ret = MPI_Wait(&request, &state);
		for (i = 0; i < nprocs * N; i++) 
			fprintf(stderr, "myrank = %d, recv_buffer[%d] = %d\n", myrank, i, recv_buffer[i]); fprintf(stderr, "\n"); 
		delete[]recv_buffer; 
		delete[]send_buffer; 
	}
	MPI_Finalize(); 
	return 0;
}


int main(int argc, char** argv)
{
	extern int cuFDK(int argc, char **argv);
	
	//return test_mpi2d(argc, argv);
	//return test(argc, argv);
	return cuFDK(argc, argv);
}
