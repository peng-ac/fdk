#ifndef __MyThread_H
#define  __MyThread_H

#include <memory> 
#include <future> 
#include <memory>
#include <chrono>
#include <assert.h>
#include <iostream>
 
class MyThread {
public:
	enum Status {
		Ready,
		Timeout,
		Deferred
	};
	static const char* GetStatusString(Status s);
	MyThread():m_bExit(true){		
	}
	virtual ~MyThread() {
		Destroy();
	}
	virtual void OnProc() {
	}
	virtual bool IsActive() const {
		bool bRtn = false;
		if (m_thread) {
			bRtn = m_thread->joinable();
			//printf("thread joinable = %d\n", bRtn ? 1 : 0);
		}
		return bRtn;
	}
	virtual inline MyThread& Create() {
		m_bExit = false;
		m_thread = std::make_shared<std::thread>(ThreadProc, this);
		return *this;
	}
	virtual inline MyThread& Destroy() {
		if (IsActive()) {
			m_bExit = true;
			if (m_thread->joinable()) {
				//printf("begin join\n");
				m_thread->join();
				//printf("end join\n");
			}
			//printf("begin m_thread.reset(),%s\n", name.c_str());
			m_thread.reset();
			//printf("end m_thread.reset(),%s\n", name.c_str());
			m_bExit = false;
		}
		return *this;
	}
	int id() const{
		return 0; 
	}
protected:
	static int ThreadProc(void* lpdwThreadParam) {
		assert(lpdwThreadParam);
		((MyThread*)lpdwThreadParam)->OnProc();
		return 0;
	}
	std::shared_ptr<std::thread> m_thread;
public:
	std::string name;
	bool m_bExit;
};


#endif //__MyThread_H
