#ifndef DATA_LAYER_HPP
#define DATA_LAYER_HPP
#include "block.hpp"

namespace fool{

template<typename Dtype>
class DataLayer{
public:
	explicit DataLayer(){}
	virtual ~DataLayer(){}
	explicit DataLayer(
			int data_total,
			std::vector<Block<Dtype>*>& p_src_data,
			std::vector<Block<Dtype>*>& p_label);
	// data_info including src_data and different type label
	// batch_size is variable
	virtual	void load_batch(
			int batch_size,
			std::vector<Block<Dtype>*>& data_info,
			std::vector<Block<Dtype>*>& label_info);
	virtual void Next(int batch_size);
	void RandomIndex();

	int m_cursor;
	int m_data_total;
	std::vector<Block<Dtype>*> m_src_data;
	std::vector<Block<Dtype>*> m_labels;
	std::vector<int> m_index;


};

}
#endif // DATA_LAYER_HPP
