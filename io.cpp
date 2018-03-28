#include "io.hpp"
#include <iostream>
#include <string>
#include <fstream>

namespace fool{

template<typename Dtype>
void MNIST<Dtype>::ReadMnistImage(std::string& path, std::vector<cv::Mat>& images){
	std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
	if(!ifs){
		std::cout << "cannot open file: "<< path  << std::endl;
	}else{
		const int numHeader = 4;
		int header[numHeader];
		std::cout << "read image header" << std::endl;
		ifs.read(reinterpret_cast<char*>(&header), sizeof(header));
		if (ifs) {
			for (int i = 0; i < numHeader; ++i) {
				header[i] = ReverseInt(header[i]);
			}
			std::cout << "read Mnist image successful." << std::endl;
			std::cout << "magic number: " << header[0] << std::endl;
			std::cout << "image number: " << header[1] << std::endl;
			std::cout << "rows  number: " << header[2] << std::endl;
			std::cout << "cols  number: " << header[3] << std::endl;
			images.resize(header[1]);
			unsigned char temp = 0;
			for (int i = 0; i < header[1]; ++i) {
				cv::Mat image = cv::Mat::zeros(header[2], header[3], CV_32FC1);
				for (int r = 0; r < header[2]; ++r) {
					for (int c = 0; c < header[3]; ++c) {
						ifs.read((char*) &temp, sizeof(temp));
						image.at<float>(r, c) = (float)temp;
					}
				}
				images[i] = image;
			}
		}
	}
	ifs.close();
}
template<typename Dtype>
void MNIST<Dtype>::ReadMnistLabel(std::string &path, std::vector<Dtype>& labels){
	std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
	if(!ifs.is_open()){
		std::cout << "read Mnist Label fail." << std::endl;
		return;
	}else{
		std::cout << "read Mnist Label successful." << std::endl;
		int magic_number = 0;
		int number_of_images = 0;
		ifs.read((char*)&magic_number, sizeof(magic_number));
		ifs.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		std::cout << "magic number = " << magic_number << std::endl;
		std::cout << "number of images = " << number_of_images << std::endl;

		for (int i = 0; i < number_of_images; i++){
			unsigned char label = 0;
			ifs.read((char*)&label, sizeof(label));
			labels.push_back((int)label);
		}
	}
}
INSTANTIATE_CLASS(MNIST);

} // namespace fool
