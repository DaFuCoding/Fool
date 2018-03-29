#ifndef FULLYCONNECT_HPP
#define FULLYCONNECT_HPP
#include "block.hpp"
#include "math.h"
#include "filter.hpp"
#include "filler_filter.hpp"

namespace fool {

template<typename Dtype>
class FullyConnectFilter: public Filter<Dtype>{
public:
	explicit FullyConnectFilter(const vector<vector<int>>& blob_shapes)
		: Filter<Dtype>(blob_shapes){
		m_K = blob_shapes[0][0];
		m_N = blob_shapes[1][0];
	}

	virtual void Reshape(const std::vector<Block<Dtype>*>& inputs,
											 const std::vector<Block<Dtype>*>& outputs);
	virtual void FilterInitialize();

	virtual void Forward_cpu(const std::vector<Block<Dtype>*>& inputs,
													 const std::vector<Block<Dtype>*>& outputs);
	virtual void Backward_cpu(const std::vector<Block<Dtype>*>& outputs,
														const std::vector<Block<Dtype>*>& inputs);

	// input C*H*W
	int m_K;
	// output C*H*W
	int m_N;
	// batch size
	int m_M;
	// bias_term in output
	Block<Dtype> m_output_bias;
};

}

#endif // FULLYCONNECT_HPP
