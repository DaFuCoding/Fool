#include "filter.hpp"

namespace fool {

template<typename Dtype>
Filter<Dtype>::Filter(const vector<vector<int>>& blob_shapes){
	m_lr_params.resize(blob_shapes.size());
	for(int i=0; i<blob_shapes.size(); ++i){
		m_lr_params[i].reset(new Blob<Dtype>());
		m_lr_params[i]->SycedBlock(blob_shapes[i]);
		m_lr_params[i].FromModel(blob_shapes[i]);
	}
}

INSTANTIATE_CLASS(Filter);

}


