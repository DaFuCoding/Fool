#ifndef MATHUTIL_HPP
#define MATHUTIL_HPP
#include "common.hpp"
#include "cblas.h"
#include "Eigen/Core"
#include "Eigen/Dense"

namespace fool {
template<typename Dtype>
void fool_set(const int N, const Dtype value, Dtype* Y);

namespace math{
// Common Eigen types that we will often use
template <typename T>
using EigenMatrixMap =
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;
template <typename T>
using EigenArrayMap =
		Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> >;
template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> >;
template <typename T>
using ConstEigenMatrixMap =
		Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;
template <typename T>
using ConstEigenArrayMap =
		Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> >;
template <typename T>
using ConstEigenVectorMap =
		Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1> >;
template <typename T>
using ConstEigenVectorArrayMap =
		Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1> >;

template<typename Dtype>
void Gemm(const CBLAS_TRANSPOSE TransA,
					const CBLAS_TRANSPOSE TransB,
					const int M,
					const int N,
					const int K,
					const float alpha,
					const float* A,
					const float* B,
					const float beta,
					float* C);
template<typename Dtype>
void Gemv(const CBLAS_TRANSPOSE TransA, const int M,
					const int N, const float alpha, const float* A, const float* x,
					const float beta, float* y);

template<typename Dtype>
void Axpy();
} // math namespace

}
#endif // MATHUTIL_HPP
