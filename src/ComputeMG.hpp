
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef COMPUTEMG_HPP
#define COMPUTEMG_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x);
#if !defined(HPCG_NOHPX)
hpx::future<int> ComputeMG_async(const SparseMatrix  & A, const Vector & r, Vector & x);
#endif

#endif // COMPUTEMG_HPP
