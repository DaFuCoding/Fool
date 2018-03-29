#include "fully_connect_filter.hpp"
#include "block.hpp"

namespace fool{

template<typename Dtype>
void FullyConnectFilter<Dtype>::FilterInitialize(){
	std::shared_ptr<FillerFilter<Dtype>> weight_filler(
				GetFiller<Dtype>("gaussian", 0.1, 0.001));
	// default use bias_term
	weight_filler->Fill(this->m_lr_params[0].get());
	std::shared_ptr<FillerFilter<Dtype>> bias_filler(
				GetFiller<Dtype>("constant", 1.0));
	bias_filler->Fill(this->m_lr_params[1].get());

}

template<typename Dtype>
void FullyConnectFilter<Dtype>::Reshape(
		const std::vector<Block<Dtype>*>& inputs,
		const std::vector<Block<Dtype>*>& outputs){
	m_M = inputs[0]->count(0, 1);
	vector<int>	output_shape = inputs[0]->shape();
	output_shape[1] = m_N;
	outputs[0]->SyncedBlock(output_shape);

	// default use bias_term
	vector<int> output_bias_shape(1, m_M)	;
	m_output_bias.SyncedBlock(output_bias_shape);
	fool_set(m_M, Dtype(1), m_output_bias.mutable_cpu_data());
}

template<typename Dtype>
void FullyConnectFilter<Dtype>::Forward_cpu(
		const std::vector<Block<Dtype>*>& inputs,
		const std::vector<Block<Dtype>*>& outputs){

}

template<typename Dtype>
void FullyConnectFilter<Dtype>::Backward_cpu(
		const std::vector<Block<Dtype>*>& outputs,
		const std::vector<Block<Dtype>*>& inputs){

}

INSTANTIATE_CLASS(FullyConnectFilter);
}

