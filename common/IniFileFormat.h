#ifndef __INI_FMT_H
#define __INI_FMT_H

#include <map>
#include <string>
#include <vector>
#include <stdio.h>
#include <assert.h>

#include <fstream>

class IniFileFormat {
public:
	IniFileFormat(std::string _strSeprator = ":") :strSeprator(_strSeprator) {
		assert(!strSeprator.empty());
	}
	IniFileFormat& SetSeprator(std::string _strSeprator = ":") {
		strSeprator = _strSeprator;
		assert(!strSeprator.empty());
		return *this;
	}
	inline void Clear() {
		section_key.clear();
	}
	inline bool Load(std::string szPath, bool bClear = true) {
		assert(!strSeprator.empty());
		if (bClear) this->Clear();
		return LoadAndSplit(szPath.c_str());
	}
	inline bool Save(std::string szPath) {
		return SaveToFile(szPath.c_str());
	}
	inline std::string Value(std::string key, std::string section = "") const {
		std::string val;
		auto it = section_key.find(section);
		if (it != section_key.end()) {
			auto ita = it->second.find(key);
			if (ita != it->second.end()) {
				val = ita->second;
			}
		}
		return val;
	}
	inline void SetValue(std::string val, std::string key, std::string section = "") {
		auto it = section_key.find(section);
		if (it == section_key.end()) {
			section_key[section].insert(std::pair<std::string, std::string>(key, val));
		}
		else {
			auto ita = it->second.find(key);
			if (ita == it->second.end()) {
				it->second.insert(std::pair<std::string, std::string>(key, val));
			}
			else {
				ita->second = val;
			}
		}
	}
	inline IniFileFormat& operator=(const IniFileFormat& ini) {
		this->section_key = ini.section_key;
		this->strSeprator = ini.strSeprator;
		return *this;
	}
	inline std::vector<std::string> GetKeys(std::string section) const {
		std::vector<std::string> vec;
		auto it = this->section_key.find(section);
		if (it != section_key.end()) {
			auto mm = it->second;
			for (auto x = mm.begin(); x != mm.end(); x++) {
				vec.push_back(x->first);
			}
		}
		return vec;
	}
	std::map<std::string, std::map<std::string, std::string>> GetSectionKey() const {
		return section_key;
	}
	void SetSectionKey(std::map<std::string, std::map<std::string, std::string>> s) {
		std::map<std::string, std::map<std::string, std::string>>::iterator it;
		for (it = s.begin(); it != s.end(); it++) {
			//std::string sec = it->first;
			//std::map<std::string, std::string>::iterator iter = it->second.begin();
			for (auto iter = it->second.begin(); iter != it->second.end(); iter++) {
				//auto key = iter->first;
				//auto val = iter->second;
				this->SetValue(iter->second, iter->first, it->first);
			}
		}
	}
private:
	inline std::streampos fileSize(const char* filePath) {
		std::streampos fsize = 0;
		std::ifstream file(filePath, std::ios::binary);

		fsize = file.tellg();
		file.seekg(0, std::ios::end);
		fsize = file.tellg() - fsize;
		file.close();
		return fsize;
	}
	inline bool SaveToFile(const char* szPath) const {
		std::cout << "Write File : " << szPath << std::endl;
		std::ofstream ofs(szPath, std::ofstream::out);
		if (!ofs.is_open()) return false;
		for (auto it0 = this->section_key.begin(); it0 != section_key.end(); it0++) {
			if (!it0->first.empty()) ofs << "[" << it0->first << "]" << std::endl;
			for (auto it1 = it0->second.begin(); it1 != it0->second.end(); it1++) {
				ofs << it1->first << " = " << it1->second << std::endl;
				ofs.flush();
			}
		}
		return true;
	}
	inline bool LoadAndSplit(const char* szPath) {
		std::string section;
		int fsz = fileSize(szPath);
		std::ifstream ifs(szPath, std::ifstream::in);
		if (!ifs.is_open()) return false;
		std::vector<char> vecBuf(fsz + 1024);
		while (!ifs.eof()) {
			char* szBuf = &vecBuf[0];
			const int size = vecBuf.size();
			ifs.getline(szBuf, size);
			if (ifs.good()) {
				char* p = strchr(szBuf, '#');
				if (p) *p = 0;
				p = strchr(szBuf, ';');
				if (p) *p = 0;
				//vec_str.push_back(szBuf);
				std::string sec, key, val;
				if (Split(szBuf, sec)) {
					section = sec;
				}
				else if (Split(szBuf, key, val)) {
					if (!key.empty()) {
						this->SetValue(val, key, section);
						//section_key[section].insert(std::pair<std::string, std::string>(key, val));
					}
				}
			}
		}
		ifs.close();
		return true;
	}
	inline bool Split(const std::string& str, std::string& section) {
		if (str.size() <= 2) return false;
		size_t pos = str.find_first_of("#");
		if (pos == 0) return false;
		pos = str.find_first_of(";");
		if (pos == 0) return false;
		size_t p0 = str.find_first_of("[");
		size_t p1 = str.find_first_of("]");
		bool bRtn = true;
		if (p1 > p0 && p0 != std::string::npos && p1 != std::string::npos) {
			section.insert(section.begin(), str.begin() + p0 + 1, str.begin() + p1);
			section = TrimRight(TrimLeft(section, " "), " ");
		}
		else {
			bRtn = false;
		}
		return bRtn;
	}
	inline bool Split(const std::string& str, std::string& key, std::string& val) const {
		key = val = "";
		if (str.size() <= 2) return false;
		size_t pos = str.find_first_of("#");
		if (pos == 0) return false;
		pos = str.find_first_of(";");
		if (pos == 0) return false;
		pos = str.find_first_of(strSeprator);
		if (pos > 0 && pos != std::string::npos) {
			key.insert(key.begin(), str.begin(), str.begin() + pos);
			val.insert(val.begin(), str.begin() + pos + 1, str.end());
			key = TrimRight(TrimLeft(key, " "), " ");  key = TrimRight(TrimLeft(key, "\t"), "\t");
			val = TrimRight(TrimLeft(val, " "), " ");  val = TrimRight(TrimLeft(val, "\t"), "\t");
		}
		return true;
	}
	inline std::string GetSeprator() const {
		return this->strSeprator;
	}
	inline std::string TrimLeft(std::string str, std::string val) const {
		if (!str.empty()) {
			for (;;) {
				std::size_t pos = str.find_first_of(val);
				if (pos == 0) {
					str.erase(pos, val.size());
				}
				else {
					break;
				}
			}
		}
		return str;
	}

	inline std::string TrimRight(std::string str, std::string val) const{
		if (!str.empty()) {
			for (std::size_t pos = 0; pos != std::string::npos;) {
				pos = str.find_last_of(val);
				if (pos + val.size() == str.size()) {
					str.erase(pos, val.size());
				}
				else {
					break;
				}
			}
		}
		return str;
	}
private:
	std::map<std::string, std::map<std::string, std::string>> section_key;
	std::string strSeprator;
};


#endif //__INI_FMT_H
