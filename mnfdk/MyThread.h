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
	typedef MyThread Self;
	MyThread():m_bExit(true){
		
	}
	virtual ~MyThread() {
		
	}
	virtual void OnProc() {
	}
	virtual bool IsActive() const {
		bool bRtn = false;
		if (m_thread) {
			bRtn = m_thread->joinable();
		}
		return bRtn;
	}
	virtual inline Self& Create() {
		m_bExit = false;
		m_thread = std::make_shared<std::thread>(ThreadProc, this);
		return *this;
	}
	virtual inline Self& Destory() {
		if (IsActive()) {
			m_bExit = true;
			if (m_thread->joinable()) {
				m_thread->join();
			}
			m_thread.reset();
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
	bool m_bExit;
};


#endif //__MyThread_H
