#include "data_layer.hpp"

namespace fool{


template<typename Dtype>
DataLayer<Dtype>::DataLayer(
		int data_total,
		std::vector<Block<Dtype>*>& p_src_data,
		std::vector<Block<Dtype>*>& p_label){
	m_cursor = 0;
	m_data_total = data_total;
	m_labels = p_label;
	m_src_data = p_src_data;
}

template<typename Dtype>
void DataLayer<Dtype>::load_batch(
		int batch_size,
		std::vector<Block<Dtype>*>& data_win,
		std::vector<Block<Dtype>*>& label_win){
	if(m_cursor + batch_size > m_data_total)
		m_cursor = 0;

	for(int id=0; id<data_win.size(); ++id){
		for(int i = 0; i< data_win[id]->count(); ++i){
			data_win[id]->m_data[i] = *(m_src_data[id]->m_data + m_cursor * data_win[id]->count() + i);
		}
		for(int j=0; j<label_win[id]->count(); ++j){
			label_win[id]->m_data[j] = *(m_labels[id]->m_data + m_cursor * label_win[id]->count() + j);

		}

		//memcpy(data_win[id]->m_data, m_src_data[id]->m_data + m_cursor * data_win[id]->count(), batch_size * data_win[id]->count());
		//memcpy(label_win[id]->m_data, m_labels[id]->m_data + m_cursor * data_win[id]->count(), batch_size * label_win[id]->count());

		//data_win[id]->m_data = m_src_data[id]->m_data + m_cursor * data_win[id]->count();
		//label_win[id]->m_data = m_labels[id]->m_data + m_cursor * label_win[id]->count();
	}
	m_cursor += batch_size;
}
// Only change cursor
template<typename Dtype>
void DataLayer<Dtype>::Next(int batch_size){
}
INSTANTIATE_CLASS(DataLayer);
} // namespace fool
