#ifndef FILLERILTER_HPP
#define FILLERILTER_HPP
#include "block.hpp"
#include <assert.h>
#include <random>

namespace fool {

template<typename Dtype>
class FillerFilter{
public:
	explicit FillerFilter() {}
	virtual ~FillerFilter() {}
	virtual void Fill(Block<Dtype>* block)=0;
protected:
	std::default_random_engine genertor;
};

template <typename Dtype>
class ConstantFiller : public FillerFilter<Dtype> {
public:
	explicit ConstantFiller(Dtype value)
			: m_value(value) {}
	virtual void Fill(Block<Dtype>* block) {
		Dtype* block_data = block->mutable_cpu_data();
		const int count = block->count();
		for(int i=0; i<count; ++i)
			block_data[i] = m_value;
	}
private:
	Dtype m_value;
};

template <typename Dtype>
class GaussianFiller : public FillerFilter<Dtype> {
public:
	explicit GaussianFiller(float mean, float sigma)
			: m_mean(mean), m_sigma(sigma) {}
	// All channel number as a vector
	virtual void Fill(Block<Dtype>* block) {
		const int count = block->count();
		Dtype* block_data = block->mutable_cpu_data();
		std::normal_distribution<float> normalEngine(m_mean, m_sigma);
		for(int i=0 ;i < count; ++i)
			*(block_data++) = normalEngine(this->genertor);
	}

private:
	float m_mean;
	float m_sigma;
};

template<typename Dtype>
FillerFilter<Dtype>* GetFiller(
		const std::string type,
		const Dtype value=0,
		const Dtype mean=0,
		const Dtype sigma=0.1){
	if (type == "constant") {
		return new ConstantFiller<Dtype>(value);
	} else if (type == "gaussian") {
		return new GaussianFiller<Dtype>(mean, sigma);
		/*
	} else if (type == "positive_unitball") {
		return new PositiveUnitballFiller<Dtype>(param);
	} else if (type == "uniform") {
		return new UniformFiller<Dtype>(param);
	} else if (type == "xavier") {
		return new XavierFiller<Dtype>(param);
	} else if (type == "msra") {
		return new MSRAFiller<Dtype>(param);
	} else if (type == "bilinear") {
		return new BilinearFiller<Dtype>(param);
	*/
	} else {
		CHECK(false) << "Unknown filler name: " << type;
	}
	return (FillerFilter<Dtype>*)(nullptr);
}

}
#endif // FILLERILTER_HPP
