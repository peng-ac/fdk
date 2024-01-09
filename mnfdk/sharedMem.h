#ifndef __SHAREDMEM_H
#define __SHAREDMEM_H
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>

#include "../common/type.h"
#include <iostream>       // std::cout
#include <future>         // std::async, std::future
#include <chrono>         // std::chrono::milliseconds
#include <list>

struct SHM {
	enum SHM_TYPE{
		SERVER = 1,
		CLIENT = 2,
		IPCKEY = 0x344378,
	};
	SHM(SHM_TYPE _type, int64 _size, int64 id = -1, const char* tok = "__shmfile") {
		FILE* fp = fopen(tok, "wb");
		if (fp) fclose(fp);
		key = ftok(tok, 65); 
		type = _type;
		segment_id = id;
		segment_size = _size;
		buffer = NULL;
		//if (type == CLIENT){
		//	assert(id != -1);
		//}
		Create();
	}
	virtual~SHM() {
		if (buffer && segment_id != -1) {
			Close(segment_id);
			segment_id = -1;
			buffer = NULL;
		}
	}
	int64 GetSize() const{
		return segment_size;
	}
private:
	bool Close(int64 id) const {
		bool bRtn = false;
		void* p_setting = (void *)shmat(id, NULL, 0);
		if (p_setting != (void *)-1) {
			shmdt(p_setting);
			shmctl(id, IPC_RMID, 0);
			std::cout << "del segment, id = " << id << std::endl;
			bRtn = true;
		}
		return bRtn;
	}
	bool Create() {
		bool bRtn = false;
		int64 id = -1;
		if (type == SERVER) {
			id = shmget(key, segment_size, 0666);
			//check if existed, delete
			if (id != -1) {
				if (!Close(id)) {
					std::cout << "delete segment error : " << id << std::endl;
				}
			}
		}
		id = shmget(key, segment_size, IPC_CREAT | 0666);
		if (id == -1) {
			segment_id = -1;
			std::cout << "create segment failed, size = " << segment_size << "bytes"
				<< ", " << segment_size / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
		} else {
			std::cout << "create segment success, size = " << segment_size << "bytes"
				<< ", " << segment_size / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
			segment_id = id;
			buffer = (char*)shmat(id, NULL, 0);
			assert(buffer);
			//memset(buffer, 0, segment_size);
			bRtn = true;
		}
		return bRtn;
	}
	SHM_TYPE type;
	int64 key, segment_id, segment_size;
public:
	char* buffer;
};




#endif //__SHAREDMEM_H