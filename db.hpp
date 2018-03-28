#ifndef DB_HPP
#define DB_HPP
#include <string>
#include <vector>
#include <fstream>

namespace fool{namespace db {

class Cursor{
public:
	Cursor(){}
	virtual ~Cursor(){}
	virtual void Next(int batch_size);
	virtual void SeekToFirst();

	int m_cursor;
};

class DB{
public:
	DB(){}
	virtual ~DB(){}
	virtual void Open(const std::string& source);
	virtual void Close();
	virtual void RandomIndex();
	std::vector<unsigned int> m_index;
	int m_data_total;
	std::fstream m_file_holder;
	virtual Cursor* NewCurosr();

};

} // namespace db
} // namespace fool
#endif // DB_HPP
