#include "convolution_filter.hpp"
#include "block.hpp"
namespace fool{

template<typename Dtype>
void ConvolutionFilterCore(const Dtype* data, Dtype* outputData, const int inputChannel, const int startIdx,
							 const int outputIdx, const int kernelRow, const int kernelCol, const int channelStep,
						   const int inputColStep, const Dtype* weights, const int weightChannelStep){
	for(int c = 0; c < inputChannel; ++c){
		for(int i = 0; i < kernelRow; ++i){
			for(int j = 0; j < kernelCol; ++j){
				float inputValue = *(data + startIdx + c*channelStep + i*inputColStep + j);
				float weightValue = *(weights + weightChannelStep + c*kernelCol*kernelRow + i*kernelCol + j);
				*(outputData + outputIdx) = inputValue * weightValue;
			}
		}
	}
}

template<typename Dtype>
void ConvolutionForward(const Block<Dtype>& inputs, Block<Dtype>& outputs, const Block<Dtype>& weightParam,
						const int outputChannel, const int kernelSizeRow, const int kernelSizeCol,
						const int stride=1, const int padSize=0){
	int batchSize = inputs.batch;
	// check batch size small than 1
	assert(batchSize >= 1);

	int inputRow = inputs.height;
	int inputCol = inputs.width;
	int inputChannel = inputs.channel;
	int outputRow = (inputRow + 2*padSize - kernelSizeRow) / stride + 1;
	int outputCol = (inputCol + 2*padSize - kernelSizeCol) / stride + 1;

	Dtype* outputData = outputs.data;
	const Dtype* weight = weightParam.data;
	int inputStepLen = inputRow * inputCol;
	int outputStepLen = outputRow * outputCol;
	int inputBlockLen = inputStepLen * inputChannel;
	int outputBlockLen = outputStepLen * outputChannel;
	for(int b=0; b < batchSize; ++b){
		for(int c = 0; c < outputChannel; ++c){
			for(int i = 0; i < outputRow; i += stride){
				for(int j = 0; j < outputCol; j += stride){
					int startIdx = i * outputCol + j;
					int outputIdx = c * outputStepLen + i * outputCol + j;
					ConvolutionFilterCore(inputs.data, outputData, inputChannel, startIdx, outputIdx, kernelSizeRow, kernelSizeCol,
										  inputStepLen, inputCol, weight, c*kernelSizeCol*kernelSizeRow );
				}
			}
		}

	}



	return;

}
template<typename Dtype>
void ConvolutionFilter<Dtype>::FilterInitize(){

}


} // namespace FOOL

