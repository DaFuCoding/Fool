#include "flatten_filter.hpp"
namespace fool {


template<typename Dtype>
void FlattenFilter<Dtype>::Reshape(const Block<Dtype>& inputs, Block<Dtype>& outputs){
	// Same dimination
	CHECK_EQ(inputs.count(), outputs.count());
	// data copy
	for(int i=0; i < inputs.count(); ++i){
		outputs.m_data[i] = inputs.m_data[i];
	}

}
template<typename Dtype>
void FlattenFilter<Dtype>::FilterInitialize(){

}

template<typename Dtype>
void FlattenFilter<Dtype>::Forward_cpu(const std::vector<Block<Dtype>*>& inputs,
																const std::vector<Block<Dtype>*>& outputs)
{

}

template<typename Dtype>
void FlattenFilter<Dtype>::Backward_cpu(const std::vector<Block<Dtype>*>& inputs,
																 const std::vector<Block<Dtype>*>& outputs)
{

}

INSTANTIATE_CLASS(FlattenFilter);

}
