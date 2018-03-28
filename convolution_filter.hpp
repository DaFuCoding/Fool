#ifndef FOOL_ConvolutionFilter_HPP
#define FOOL_ConvolutionFilter_HPP
#include <opencv2/opencv.hpp>
#include <iostream>
#include "block.hpp"
#include "filter.hpp"

namespace fool{

template<typename Dtype>
class ConvolutionFilter : public Filter<Dtype>{
public:
	explicit ConvolutionFilter(){}

	void ConvolutionForward(const Block<Dtype>& inputs, Block<Dtype>& outputs, const Block<Dtype>& weightParam,
							const int outputChannel, const int kernelSizeRow, const int kernelSizeCol,
							const int stride=1, const int padSize=0);

	void ConvolutionFilterCore(const Dtype* data, Dtype* outputData, const int inputChannel, const int startIdx,
							   const int outputIdx, const int kernelRow, const int kernelCol, const int channelStep,
							   const int inputColStep, const Dtype* weights, const int weightChannelStep);
	virtual void FilterInitialize();
};

}


#endif // FOOL_ConvolutionFilter_HPP
