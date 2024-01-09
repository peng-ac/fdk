#ifndef __LOGGER_H
#define __LOGGER_H


#ifdef _WIN32
#pragma warning(disable : 4189 4100 4267)
#endif

#include <time.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <string>
#include <iostream>     
#include <iomanip>      
#include <iosfwd>
#include <sstream>
#include <iostream>
#include <fstream>      
#include <mutex>


#ifndef _WIN32
#define _vsnprintf vsnprintf
#endif

struct Logger {
	enum eLevel {
		LevelDebug,
		LevelWarning,
		LevelCritical,
		LevelFatal,
		LevelOff
	};
	Logger(const char* pLogFilePath = "./log.txt", int _nLevel = LevelDebug, int _nMaxFileSize = 1024 * 1024 * 8)
		:nLevel(_nLevel), sLogFilePath(pLogFilePath), nMaxFileSize(_nMaxFileSize), m_fp(NULL) {
		static const char szEndl[] = "\n";
		endl = szEndl;
		this->Open();
	}
	virtual ~Logger() {
		if (m_fp) fclose(m_fp); m_fp = NULL;
	}
	inline Logger& SetLevel(int level) {
		nLevel = level;
		return *this;
	}
	inline Logger& Printf(const char* fmt, ...) {
		if (nCurrentLevel < nLevel) return *this;
		char szBuf[1024 * 8] = "";
		va_list list;
		va_start(list, fmt);
		_vsnprintf(szBuf, sizeof(szBuf), fmt, list);
		va_end(list);
		_Printf(szBuf);
		return *this;
	}
	inline Logger&  PrintfTime() { return PrintfWidthTime(""); }
	inline Logger&  PrintfWidthTime(const char* fmt, ...) {
		if (nCurrentLevel < nLevel) return *this;
		char szBuf[1024 * 8] = "";
		va_list list;
		va_start(list, fmt);
		_vsnprintf(szBuf, sizeof(szBuf), fmt, list);
		va_end(list);
		return Printf("%s\t%s", TimeString(), szBuf);
	}
	inline std::string Format(const char* fmt, ...) const {
		char szBuf[1024 * 8] = "";
		va_list list;
		va_start(list, fmt);
		_vsnprintf(szBuf, sizeof(szBuf), fmt, list);
		va_end(list);
		return std::string(szBuf);
	}
	template<typename T> Logger& operator<<(T val) {
		if (nCurrentLevel < nLevel) return *this;
		else                        return this->_Out(val);
	}
#ifdef _WIN32
	template<> Logger& operator<<<std::ostringstream>(std::ostringstream ostr) {
		return (*this) << ostr.str().c_str();
	}
#endif
	inline const char* TimeString() {
		time_t atime, rtime = time(&atime);
		struct tm * strtim = localtime(&rtime);
		szTime[0] = 0;
		if (strtim) snprintf(szTime, sizeof(szTime), "%04d:%02d:%02d %02d:%02d:%02d", strtim->tm_year + 1900, strtim->tm_mon + 1, strtim->tm_mday, strtim->tm_hour, strtim->tm_min, strtim->tm_sec);
		return szTime;
	}
	inline  std::string GetUniqueTime() {
		time_t atime, rtime = time(&atime);
		struct tm * strtim = localtime(&rtime);
		char szTxt[1024] = "";
		snprintf(szTxt, sizeof(szTxt), "%04d_%02d_%02d-%02d_%02d_%02d", strtim->tm_year + 1900, strtim->tm_mon + 1, strtim->tm_mday, strtim->tm_hour, strtim->tm_min, strtim->tm_sec);
		return std::string(szTxt);
	}
	inline long FileSize(FILE* fp) const {
		if (fp == 0) return 0;
		long nsize = 0;
		long cur = ftell(fp);
		if (0 == fseek(fp, 0, SEEK_END)) {
			nsize = ftell(fp);
			fseek(fp, cur, SEEK_SET);
		}
		return nsize;
	}
	inline long FileSize(const char* pszPath) {
		FILE* fp = fopen(pszPath, "rb");
		if (fp == 0) return 0;
		long nsize = FileSize(fp);
		fclose(fp);
		return nsize;
	}
	inline Logger&  Loggerput(const char* fmt, ...) {
		const char LvName[][10] = {
			"INFO"
			,"WARNING"
			,"CRITICAL"
			,"FATAL"
			,"-"
		};
		//		if (nCurrentLevel < nLevel) return *this;
		char szBuf[1024 * 8] = "";
		va_list list;
		va_start(list, fmt);
		vsnprintf(szBuf, sizeof(szBuf), fmt, list);
		va_end(list);
		return Printf("%s\t[%s]\t%s", TimeString(), LvName[nCurrentLevel], szBuf);
	}

	int   nCurrentLevel;
	const char* endl;
protected:
	int nMaxFileSize;
	std::string sLogFilePath;
	char  szTime[96];
	int   nLevel;
	inline bool Open() {
		bool bRtn = false;
		if (!m_fp) {
			extern void MakePathDirIfNeed(const char* sz);
			MakePathDirIfNeed(sLogFilePath.c_str());
			if (FileSize(sLogFilePath.c_str()) < nMaxFileSize) {
				m_fp = fopen(sLogFilePath.c_str(), "at");
			}
			else {
				std::string sLogFilePathBak = sLogFilePath + "." + GetUniqueTime();
				std::remove(sLogFilePathBak.c_str());
				std::rename(sLogFilePath.c_str(), sLogFilePathBak.c_str());
				m_fp = fopen(sLogFilePath.c_str(), "wt");
			}
		}
		return m_fp ? true : false;
	}
	inline Logger& _Printf(const char* szText) {
		std::lock_guard<std::mutex> _lg(mtx);
		try {
			if (this->Open()) {
				fprintf(m_fp, szText);
				fflush(m_fp);
			}
		}
		catch (...) {
		}
		return *this;
	}
	template<typename T> inline Logger& _Out(T& val) {
		std::ostringstream ostr; ostr << val; _Printf(ostr.str().c_str()); return *this;
	}
private:
	std::mutex mtx;
	FILE* m_fp;
};

#endif