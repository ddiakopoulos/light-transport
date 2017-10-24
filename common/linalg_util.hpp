// This is free and unencumbered software released into the public domain.
// For more information, please refer to <http://unlicense.org>

#ifndef linalg_util_h
#define linalg_util_h

#include "linalg.h"
#include <iostream>

using namespace linalg::aliases;

static const float4x4 Identity4x4 = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
static const float3x3 Identity3x3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
static const float2x2 Identity2x2 = {{1, 0}, {0, 1}};
    
static const float4x4 Zero4x4 = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
static const float3x3 Zero3x3 = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
static const float2x2 Zero2x2 = {{0, 0}, {0, 0}};

template<class T, int M> linalg::vec<T, M>  safe_normalize(const linalg::vec<T,M> & a)  { return a / std::max(T(1E-6), length(a)); }
template<class T, int N> linalg::mat<T,N,N> inv(const linalg::mat<T,N,N> & a)           { return inverse(a); }

template<class T> std::ostream & operator << (std::ostream & a, const linalg::vec<T,2> & b) { return a << '{' << b.x << ", " << b.y << '}'; }
template<class T> std::ostream & operator << (std::ostream & a, const linalg::vec<T,3> & b) { return a << '{' << b.x << ", " << b.y << ", " << b.z << '}'; }
template<class T> std::ostream & operator << (std::ostream & a, const linalg::vec<T,4> & b) { return a << '{' << b.x << ", " << b.y << ", " << b.z << ", " << b.w << '}'; }

template<class T, int N> std::ostream & operator << (std::ostream & a, const linalg::mat<T,2,N> & b) { return a << '\n' << b.row(0) << '\n' << b.row(1) << '\n'; }
template<class T, int N> std::ostream & operator << (std::ostream & a, const linalg::mat<T,3,N> & b) { return a << '\n' << b.row(0) << '\n' << b.row(1) << '\n' << b.row(2) << '\n'; }
template<class T, int N> std::ostream & operator << (std::ostream & a, const linalg::mat<T,4,N> & b) { return a << '\n' << b.row(0) << '\n' << b.row(1) << '\n' << b.row(2) << '\n' << b.row(3) << '\n'; }

// Bias matrix is a constant.
// It performs a linear transformation to go from the [–1, 1]
// range to the [0, 1] range. Having the coordinates in the [0, 1]
// range is necessary for the values to be used as texture coordinates.
// projector = bias * projection * view * model
static const float4x4 texcoord_bias_matrix = {
    { 0.5f,  0.0f,  0.0f,  0.0f },
    { 0.0f,  0.5f,  0.0f,  0.0f },
    { 0.0f,  0.0f,  0.5f,  0.0f },
    { 0.5f,  0.5f,  0.5f,  1.0f }
};

#endif