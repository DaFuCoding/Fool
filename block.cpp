#include "block.hpp"

namespace fool{

template<typename Dtype>
Block<Dtype>::Block(const std::vector<int>& shape){
	m_capacity = 0;
	SyncedBlock(shape);
}

template<typename Dtype>
void Block<Dtype>::SyncedBlock(const std::vector<int>& shape){
	m_count = 1;
	m_shape.resize(shape.size());
	for(unsigned int i = 0; i < shape.size(); ++i){
		m_count *= shape[i];
		m_shape[i] = shape[i];
	}
	if(m_count > m_capacity){
		m_capacity = m_count;
		m_data.reset(new MemoryController(m_capacity * sizeof(Dtype)));
		m_diff.reset(new MemoryController(m_capacity * sizeof(Dtype)));
	}
}
// static_cast don't adapt to const-ness
template<typename Dtype>
const Dtype* Block<Dtype>::cpu_data(){
	return (const Dtype*)(m_data->cpu_data());
}

template<typename Dtype>
Dtype* Block<Dtype>::mutable_cpu_data(){
	return static_cast<Dtype*>(m_data->mutable_cpu_data());
}
template<typename Dtype>
const Dtype* Block<Dtype>::cpu_diff(){
	return (const Dtype*)(m_diff->cpu_data());
}
template<typename Dtype>
Dtype* Block<Dtype>::mutable_cpu_diff(){
	return static_cast<Dtype*>(m_diff->mutable_cpu_data());
}

template<typename Dtype>
void Block<Dtype>::FromModel(vector<int>& shape){

}

INSTANTIATE_CLASS(Block);
// label information
template class Block<int>;
template class Block<unsigned int>;

}
