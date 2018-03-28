#ifndef FLATTEN_HPP
#define FLATTEN_HPP

#include "block.hpp"
#include "filter.hpp"

namespace fool {

template <typename Dtype>
class FlattenFilter : public Filter<Dtype>{
public:
	explicit FlattenFilter() : Filter<Dtype>() {}
	void Reshape(const Block<Dtype>& inputs, Block<Dtype>& outputs);

	virtual void FilterInitialize();
	virtual void Forward_cpu(const std::vector<Block<Dtype>*>& inputs,
													 const std::vector<Block<Dtype>*>& outputs);
	virtual void Backward_cpu(const std::vector<Block<Dtype>*>& inputs,
														const std::vector<Block<Dtype>*>& outputs);

};
}
#endif // FLATTEN_HPP
