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

	explicit FullyConnectFilter(const int input_dim, const int output_dim):m_K(input_dim){
		m_K = input_dim;
		m_N = output_dim;
		vector<int> input_vec = {input_dim};
		vector<int> output_vec = {output_dim};
		vector<vector<int>> temp_blob_shapes;
		temp_blob_shapes.push_back(input_vec);
		temp_blob_shapes.push_back(output_vec);
//		FullyConnectFilter<Dtype>(temp_blob_shapes);
	}

	void FilterSetUp(const std::vector<Block<Dtype>*>& inputs,
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
};

}

#endif // FULLYCONNECT_HPP
