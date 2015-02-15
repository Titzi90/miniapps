
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

#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeSYMGS( const SparseMatrix  & A, const Vector & x, Vector & y);
#ifndef NO_HPX
int ComputeSYMGS_sub_async(const SparseMatrix  & A, const Vector & x, Vector & y);
int ComputeSYMGS_sub_async_twostep(const SparseMatrix  & A, const Vector & x, Vector & y);
int ComputeSYMGS_sub_async_twostep_revers(const SparseMatrix  & A, const Vector & x, Vector & y);
#endif


#endif // COMPUTESYMGS_HPP
