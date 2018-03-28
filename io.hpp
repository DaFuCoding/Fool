#ifndef FOOL_IO_HPP
#define FOOL_IO_HPP
#include <vector>
#include <opencv2/opencv.hpp>
#include "common.hpp"
#include "block.hpp"

namespace fool{

template<typename Dtype>
class MNIST{
public:
	void ReadMnistImage(std::string& path, std::vector<cv::Mat>&);
	void ReadMnistLabel(std::string& path, std::vector<Dtype>&);
	inline int ReverseInt(int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255;
		c2 = (i >> 8) & 255;
		c3 = (i >> 16) & 255;
		c4 = (i >> 24) & 255;
		return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
	}
};

template<typename Dtype>
void MatToBlock(
		const std::vector<cv::Mat>& images,
		Block<Dtype>& datum){
	CHECK_EQ(datum.m_shape[0], images.size());
	int image_num = images.size();
	int channel = images[0].channels();
	int width = images[0].cols;
	int height = images[0].rows;
	int image_size = channel * width * height;
	// Need a large memory to save train data
	// datum keep the same size
	for(int i=0; i<image_num; ++i){
		for(int k=0; k<channel; ++k)
			for(int r=0; r<height; ++r)
				for(int c = 0; c < width; ++c)
					datum.m_data[i*image_size + r*height + c] = images[i].at<float>(r, c);
	}

}

template<typename Dtype>
void VectorToBlock(const std::vector<Dtype>& vecs,
									 Block<Dtype>& datum){
	for(int i=0; i<vecs.size(); ++i){
		datum.m_data[i] = vecs[i];
	}

}
} // namespace fool

#endif // FOOL_IO_HPP
