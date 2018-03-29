#ifndef FILTER_HPP
#define FILTER_HPP
#include "block.hpp"

namespace fool{

// Abstract class
template<typename Dtype>
class Filter{

public:
	Filter(){}

	explicit Filter(const vector<vector<int>>& blob_shapes);
	virtual ~Filter(){}
	virtual	void FilterInitialize()=0;

	inline Dtype Forward(const vector<Block<Dtype>*>& inputs,
											const vector<Block<Dtype>*>& outputs);
	inline Dtype Backward(const vector<Block<Dtype>*>& outputs,
											const vector<Block<Dtype>*>& inputs);

	virtual void Forward_cpu(const vector<Block<Dtype>*>& inputs,
													 const vector<Block<Dtype>*>& outputs) = 0;
	virtual void Backward_cpu(const vector<Block<Dtype>*>& outputs,
														const vector<Block<Dtype>*>& inputs) = 0;

	vector<shared_ptr<Block<Dtype>>> m_lr_params;
	vector<Dtype> m_loss;
	string m_initial_type;

};
}
#endif // FILTER_HPP
