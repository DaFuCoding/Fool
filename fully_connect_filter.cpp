#include "fully_connect_filter.hpp"
#include "block.hpp"

namespace fool{

template<typename Dtype>
void FullyConnectFilter<Dtype>::FilterInitialize(){
	std::vector<int> lr_w_shape = {m_K, m_N};
	this->m_lr_params[0] = new Block<Dtype>(lr_w_shape);
	//std::shared_ptr<FillerFilter<Dtype>> weight_filler(new GaussianFiller<Dtype>(0.1, 0.001));
	FillerFilter<Dtype>* weight_filler(new GaussianFiller<Dtype>(0.1, 0.001));
	weight_filler->Fill(this->m_lr_params[0]);
	delete weight_filler;
	weight_filler = nullptr;

	std::vector<int> lr_bias_shape = {1, m_N};
	this->m_lr_params[1] = new Block<Dtype>(lr_bias_shape);
	std::shared_ptr<FillerFilter<Dtype>> bias_filler(new ConstantFiller<Dtype>(0.0));
	bias_filler->Fill(this->m_lr_params[1]);

}

template<typename Dtype>
void FullyConnectFilter<Dtype>::FilterSetUp(
		const std::vector<Block<Dtype>*>& inputs,
		const std::vector<Block<Dtype>*>& outputs){
	m_K =	inputs[0]->count(1);
	m_N = outputs[0]->count(1);
	// default use bias_term
	this->m_lr_params.resize(2);
	FilterInitialize();

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

