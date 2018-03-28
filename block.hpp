#ifndef FOOL_BLOCK_HPP
#define FOOL_BLOCK_HPP
#include <stdlib.h>
#include <algorithm>
#include <string>
#include "common.hpp"
#include "memory_controller.hpp"
#include <memory>

namespace fool{

template<typename Dtype>
class Block{
public:
	Block(): m_count(0),m_capacity(0), m_shape(), m_data(){}

	explicit Block(const vector<int>& shape);
	void SyncedBlock(const vector<int>& shape);
	void SyncedBlock(const Block<Dtype>& other){
		SyncedBlock(other.m_shape);
	}
	virtual ~Block(){}

	// Shape information
	inline int num_axes() const { return m_shape.size(); }
	inline int count() const { return m_count; }
	inline string shape_string() const {
		ostringstream stream;
		for (int i = 0; i < m_shape.size(); ++i) {
			stream << m_shape[i] << " ";
		}
		stream << "(" << m_count << ")";
		return stream.str();
	}
	inline int count(int start_axis, int end_axis=-1) const {
		int count = 1;
		CHECK_GE(start_axis, 0);
		if(end_axis == -1)
			end_axis = m_shape.size();
		CHECK_LE(start_axis, end_axis);
		CHECK_GE(end_axis, 0);
		CHECK_LE(start_axis, num_axes());
		CHECK_LE(end_axis, num_axes());

		for (int i = start_axis; i < end_axis; ++i) {
			count *= m_shape[i];
		}
		return count;
	}
	inline int offset(const int n, const int c = 0, const int h = 0,
			const int w = 0) const {
		CHECK_GE(n, 0);
		CHECK_LE(n, m_shape[0]);
		CHECK_GE(m_shape[1], 0);
		CHECK_LE(c, m_shape[1]);
		CHECK_GE(m_shape[2], 0);
		CHECK_LE(h, m_shape[2]);
		CHECK_GE(m_shape[3], 0);
		CHECK_LE(w,m_shape[3]);
		return ((n * m_shape[1] + c) * m_shape[2] + h) * m_shape[3] + w;
	}

	// Get Xpu data
	const Dtype* cpu_data();
	Dtype* mutable_cpu_data();
	const Dtype* cpu_diff();
	Dtype* mutable_cpu_diff();
	// Get data from mdoel
	void FromModel(vector<int>& shape);

protected:
	int m_count;
	int m_capacity;
	vector<int> m_shape;
	shared_ptr<MemoryController> m_data; // NCHW
	shared_ptr<MemoryController> m_diff;

};

}
#endif // FOOL_BLOCK_HPP
