#include "block.hpp"
#include "math_util.hpp"

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

namespace math{
// General matrix multiplication
#ifdef USE_EIGEN_FOR_BLAS
template<>
void Gemm<float>(
		const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB,
		const int M,
		const int N,
		const int K,
		const float alpha,
		const float* A,
		const float* B,
		const float beta,
		float* C
		){
	// Like caffe2/utils/math_cpu.cc
	auto C_mat = EigenMatrixMap<float>(C, N, M);
	if (beta == 0) {
		C_mat.setZero();
	} else {
		C_mat *= beta;
	}
	switch (TransA) {
	case CblasNoTrans: {
		switch (TransB) {
		case CblasNoTrans:
			C_mat.noalias() += alpha * (
					ConstEigenMatrixMap<float>(B, N, K) *
					ConstEigenMatrixMap<float>(A, K, M));
			return;
		case CblasTrans:
			C_mat.noalias() += alpha * (
					ConstEigenMatrixMap<float>(B, K, N).transpose() *
					ConstEigenMatrixMap<float>(A, K, M));
			return;
		default:
			LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB";
		}
	}
	case CblasTrans: {
		switch (TransB) {
		case CblasNoTrans:
			C_mat.noalias() += alpha * (
					ConstEigenMatrixMap<float>(B, N, K) *
					ConstEigenMatrixMap<float>(A, M, K).transpose());
			return;
		case CblasTrans:
			C_mat.noalias() += alpha * (
					ConstEigenMatrixMap<float>(B, K, N).transpose() *
					ConstEigenMatrixMap<float>(A, M, K).transpose());
			return;
		default:
			LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB";
		}
	}
	default:
		LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransA";
	}
}

template<>
void Gemv<float>(
		const CBLAS_TRANSPOSE TransA,
		const int M,
		const int N,
		const float alpha,
		const float* A,
		const float* x,
		const float beta,
		float* y){
	// Like caffe2/utils/math_cpu.cc
	EigenVectorMap<float> y_vec(y, TransA == CblasNoTrans ? M : N);
		if (beta == 0) {
			// In Caffe2 we often do a lazy initialization, which may contain NaNs in
			// the float values. As a result, if beta is 0, we explicitly do a setzero.
			y_vec.setZero();
		} else {
			y_vec *= beta;
		}
		switch (TransA) {
			case CblasNoTrans: {
				y_vec.noalias() += alpha * (
						ConstEigenMatrixMap<float>(A, N, M).transpose() *
						ConstEigenVectorMap<float>(x, N));
				return;
			}
			case CblasTrans: {
				y_vec.noalias() += alpha * (
						ConstEigenMatrixMap<float>(A, N, M) *
						ConstEigenVectorMap<float>(x, M));
				return;
			}
			default:
				LOG(FATAL) << "Gemv float found an unexpected CBLAS_TRANSPOSE input.";
		}
}


template<>
void Axpy<float>(){

}
#else

#endif
} // math namespace
} // fool namespace
