#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "io.hpp"
#include "block.hpp"
#include "convolution_filter.hpp"
#include "filler_filter.hpp"
#include "registry.hpp"
#include "common.hpp"
#include "flatten_filter.hpp"
#include "fully_connect_filter.hpp"
#include "data_layer.hpp"

using namespace std;

namespace fool {

void TEST_Block(){
	// Create Block
	vector<int> shape = {1,2,3,4};
	Block<int> block(shape);
	int* data =	block.mutable_cpu_data();
	cout <<	block.shape_string() << endl;
	for(int i=0;i<block.count();++i){
		data[i] = i+1;
	}
	const int* datum = block.cpu_data();
	for(int i=0;i<block.count();++i){
		cout << datum[i] << endl;
	}


}

vector<vector<int>> ConstrcutShape(const int input_dim, const int output_dim){
	vector<vector<int>> blob_shapes;
	vector<int> blob_one({input_dim, output_dim});
	vector<int> blob_two(1, output_dim);
	blob_shapes.push_back(blob_one);
	blob_shapes.push_back(blob_two);
	return blob_shapes;
}

void TEST_Filler(){
	std::vector<int> shape = {10,1,1,1};
	shared_ptr<Block<float>> weights(new Block<float>(shape));
	int numbers = 10;
	float value = 1.414;
	float mean = 0;
	float sigma = 0.1;

	// constant type
	shared_ptr<FillerFilter<float>> const_generator(
				GetFiller<float>("constant", value));
	const_generator->Fill(weights.get());
	const float* data = weights->cpu_data();
	for(int i=0;i<numbers;++i)
		cout << data[i] << ' ';
	cout<< endl;

	// guassian type
	shared_ptr<FillerFilter<float>> normal_generator(
				GetFiller<float>("gaussian", 0, mean, sigma));
	normal_generator->Fill(weights.get());
	const float* data_normal = weights->cpu_data();
	for(int i=0;i<numbers;i++)
		cout <<*(data_normal+i) << ' ';
	cout << endl;
}
/*
void TEST_Conv(const cv::Mat& image){
	ConvolutionFilter convFilter;
	std::vector<int> inputs_shape = {1, image.channels(), image.rows, image.cols};
	Block<float> *inputs_block = new Block<float>();
	inputs_block->SyncedBlock(inputs_shape);
//	TEST_Block(image, *inputs_block);

//	convFilter.ConvolutionForward(inputs, Block<Dtype>& outputs, const Block<Dtype>& weightParam,
//								  const int outputChannel, const int kernelSizeRow, const int kernelSizeCol,
//								  const int stride=1, const int padSize=0);

}
*/
/*
void TEST_Random_Data(){

	std::default_random_engine generator;
	std::normal_distribution<float> normalEngine(0, 0.1);
	vector<float> bins;
	map<float,int> counts;
	for(int i = 0; i <= 10; i++){
		float value = (i - 10)/10.;
		bins.push_back(value);
	}
	for(int i=0; i < 300; ++i){
		float value = normalEngine(generator);
		cout << value << ' ';
		for(int c = 0; c < bins.size(); ++c){
			if(bins[c] > value){
				counts[bins[c]]++;
				cout << endl;
				break;
			}
		}
	}
	for(float i = 0; i < bins.size(); i++){
		cout << bins[i] <<' ';
		for(int n = 0; n < counts[bins[i]]; ++n)
			cout << '*';
		cout << endl;
	}
}

void TEST_Read_MNIST(std::string m_datadir){
	MNIST<float> mnist;
	std::string mnist_val_image_path = m_datadir + "/t10k-images-idx3-ubyte";
	std::string mnist_val_label_path = m_datadir + "/t10k-labels-idx1-ubyte";
	std::vector<cv::Mat> images;
	std::vector<float> labels;
	mnist.ReadMnistImage(mnist_val_image_path, images);
	mnist.ReadMnistLabel(mnist_val_label_path, labels);

	int image_total_num = 10000;
	int batch_size = 64;
	Block<float> data_total, label_total;
	std::vector<int> input_data_total_shape = {image_total_num, 1, 28,28};
	std::vector<int> input_label_total_shape = {image_total_num, 10, 1, 1};
	data_total.SyncedBlock(input_data_total_shape);
	label_total.SyncedBlock(input_label_total_shape);
	MatToBlock(images, data_total);
	VectorToBlock(labels, label_total);
	std::vector<Block<float>*> multi_data_total;
	std::vector<Block<float>*> multi_label_total;
	multi_data_total.push_back(&data_total);
	multi_label_total.push_back(&label_total);
	DataLayer<float> data_prefetch(image_total_num, multi_data_total, multi_label_total);

	std::vector<Block<float>*> data_wins;
	std::vector<Block<float>*> label_wins;
	std::vector<int> input_data_shape = {batch_size, 1, 28, 28};
	std::vector<int> input_label_shape = {batch_size, 10};
	Block<float> data_win(input_data_shape);
	Block<float> label_win(input_label_shape);
	data_wins.push_back(&data_win);
	label_wins.push_back(&label_win);

	// network model FC(784, 1000) FC(1000, 10)
	FullyConnectFilter<float>	fc1(784, 1000);
	std::vector<int> fc1_output_shape = {batch_size, 1000};
	std::vector<Block<float>*> fc1_output;
	Block<float> fc1_output_block;
	fc1_output_block.SyncedBlock(fc1_output_shape);
	fc1_output.push_back(&fc1_output_block);
	fc1.FilterSetUp(multi_data_total, fc1_output);
	FullyConnectFilter<float> fc2(1000, 10);
	std::vector<int> fc2_output_shape = {batch_size, 10};
	std::vector<Block<float>*> fc2_output;
	Block<float> fc2_output_block;
	fc2_output_block.SyncedBlock(fc2_output_shape);
	fc2_output.push_back(&fc2_output_block);
	fc2.FilterSetUp(multi_data_total, fc2_output);
	for(int epoch=0; epoch < 1; ++epoch){
		//for(int i = 0 ; i <= image_total_num/batch_size; ++i){
		for(int step =0 ;step < 1; ++step){
			data_prefetch.load_batch(batch_size, data_wins, label_wins);
		}
	}

}
void TEST_RUN(){

}

void TEST_Flatten(){
	Block<float> inputs(std::vector<int>({1,2,2,2}));
	for(int i=0;i<2*2*2;i++)
		inputs.m_data[i] = i+1;
	Block<float> outputs(std::vector<int>({1,8,1,1}));
	FlattenFilter<float> flatten;
	flatten.Reshape(inputs, outputs);
	for(int i=0;i<2*2*2;i++)
		cout << outputs.m_data[i]<<' ';
	cout  << outputs.shape_string()<< endl;
}
*/
void TEST_FullyConnect(){
	Block<float>* fc1_input_block(new Block<float>(std::vector<int>({5, 784})));
	shared_ptr<FillerFilter<float>> fill_gen(GetFiller<float>("constant", 2));
	fill_gen->Fill(fc1_input_block);
	Block<float>* fc1_nobatch_input_block(new Block<float>(vector<int>({1, 784})));
	Block<float>* fc1_output_block(new Block<float>());

	std::vector<Block<float>*> fc1_inputs;
	std::vector<Block<float>*> fc1_outputs;
	fc1_inputs.push_back(fc1_input_block);
	fc1_outputs.push_back(fc1_output_block);

	// network model FC(784, 1000) FC(1000, 10)
	FullyConnectFilter<float>	fc1(ConstrcutShape(784, 10));
	fc1.FilterInitialize();

	bool check_param = false;
	if(check_param){
		// check lr_param
		const float* fc1_bias_data = fc1.m_lr_params[1]->cpu_data();
		const float* fc1_matrix_data = fc1.m_lr_params[0]->cpu_data();
		for(int i=0;i<10;i++){
			//cout << fc1_bias_data[i] << ' ';
			for(int j=0;j<784;++j){
				cout << fc1_matrix_data[i*784+j] << ' ';
			}
			cout << endl;
		}
	}
	fc1.Reshape(fc1_inputs, fc1_outputs);
	fc1.Forward_cpu(fc1_inputs, fc1_outputs);

	delete fc1_input_block;
	delete fc1_nobatch_input_block;
	delete fc1_output_block;
}

void TEST_Memory(){
	float* p = new float();
	delete p;
	p = NULL;
	float* m = (float*)alloca(sizeof(float));
	//cout <<	sizeof(*m) << endl;

	float* nm = new(m) float();
	cout <<	sizeof(*new(m) float()) << endl;
	return;
}
void TEST_DataLayer(){

}

} // FOOL namespace

class A{
public:
	A(){
		data = (int*)malloc(10 * sizeof(int));
	}
	~A(){
		if(data != nullptr)
			free(data);
	}

	int* data;
};

int main(int argc, char *argv[]){
	//fool::TEST_Block();
	//fool::TEST_Filler();
	fool::TEST_FullyConnect();

	//fool::TEST_Random_Data();
//	fool::TEST_Flatten();
//	fool::TEST_Memory();
	//fool::TEST_DataLayer();
	//fool::TEST_Read_MNIST("/home/dafu/data/MNIST");
	//fool::TEST_FullyConnect();


	return 0;
}
