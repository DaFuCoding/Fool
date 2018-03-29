#include "block.hpp"

namespace fool {

template<typename Dtype>
void fool_set(const int N, const Dtype value, Dtype* Y){
	if(value == 0)	{
		memset(Y, 0, N);
		return;
	}
	for(int i=0; i<N; ++i)
		Y[i] = value;
}
template void fool_set<int>(const int N, const int value,int* Y);
template void fool_set<float>(const int N, const float value, float* Y);
template void fool_set<double>(const int N, const double value, double* Y);




// General matrix multiplication
#ifdef USE_EIGEN_FOR_BLAS
template<>
void Gemm<float>(){

	//int lda = (TransA == CblasNoTrans) ? K : M;
	//int ldb = (TransB == CblasNoTrans) ? N : K;
	//cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
	//						ldb, beta, C, N);
}

template<>
void Gemv<float>(
		const CBLAS_TRANSPOSE TransA, const int M,
		const int N, const float alpha, const float* A, const float* x,
		const float beta, float* y){

}

template<>
void Axpy<float>(){

}
#else

#endif

}
