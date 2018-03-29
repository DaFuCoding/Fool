#ifndef MATHUTIL_HPP
#define MATHUTIL_HPP
#include "common.hpp"

namespace fool {
template<typename Dtype>
void fool_set(const int N, const Dtype value, Dtype* Y);

typedef bool CBLAS_TRANSPOSE;

template<typename Dtype>
void Gemm();

template<typename Dtype>
void Gemv(const CBLAS_TRANSPOSE TransA, const int M,
					const int N, const float alpha, const float* A, const float* x,
					const float beta, float* y);

template<typename Dtype>
void Axpy();

}
#endif // MATHUTIL_HPP
