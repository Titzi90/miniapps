
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

/*!
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include "OptimizeProblem.hpp"

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
#if defined(HPCG_NOHPX)

int ComputeSPMV(const SparseMatrix & A, Vector & x, Vector & y) {

  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  A.isSpmvOptimized = false;
  return(ComputeSPMV_ref(A, x, y));
}

#else

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/lcos/when_all.hpp>
#include <algorithm>
#include <tuple>

#include <boost/iterator/counting_iterator.hpp>

/*
// async without sub domains by Harald
hpx::future<void> ComputeSPMV_async( const SparseMatrix & A, [>const<] Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NOMPI
    ExchangeHalo(A,x);
#endif

  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

  typedef boost::counting_iterator<local_int_t> iterator;

  return hpx::parallel::for_each(
    hpx::parallel::task, iterator(0), iterator(nrow),
    [xv, yv, &A](local_int_t i) {
      double sum = 0.0;
      const double * const cur_vals = A.matrixValues[i];
      const local_int_t * const cur_inds = A.mtxIndL[i];
      const int cur_nnz = A.nonzerosInRow[i];

      for (int j=0; j< cur_nnz; j++)
        sum += cur_vals[j]*xv[cur_inds[j]];
      yv[i] = sum;
    });
}
*/

/******************************************************************************/

//TODO ref und const überprüfen

// Matrix Vector Mul of one row
inline double mul(double const * mtxVal,
                  int const * mtxInd,
                  char const nonzeros,
                  double const * vecVal)
{
    double sum = 0;
    for(char i=0; i<nonzeros; ++i){
        sum += mtxVal[i] * vecVal[ mtxInd[i] ];
    }
    return sum;
}

// Matrix Vector Mul of a sub vector
// vecVal is the whole localaty vector
// subMtx and subYv are just the subVectors
// subYv = SubMtx * vecVal
inline SubVectorValues subMul(SubMatrix const & subMtx,
                              double const * vecVal,
                              SubVectorValues subYv)   //TODO warum keine Ref.????
{
    for (int row=0; row<subYv.size(); ++row)
    {
        *subYv[row] = mul(subMtx.values[row],
                          subMtx.indLoc[row],
                          *subMtx.nonzerosInRow[row],
                          vecVal);
    }
    return subYv;
}

// Sparse Matric Vector Multiplication
// makeing use of the SubDomains
int ComputeSPMV_sub_async(SparseMatrix const & A, Vector const & x, Vector& y )
{
#ifdef HPCG_DEBUG
    std::cout << "using SPMV_sub_async" << std::endl;
#endif
//TODO more localytsy

    std::vector<SubDomain>  & subAs =
        *static_cast<std::vector<SubDomain>* >(A.optimizationData);
    std::vector<SubVector>  & subXs =
        *static_cast<std::vector<SubVector>* >(x.optimizationData);
    std::vector<SubVector>& subYs =
        *static_cast<std::vector<SubVector>* >(y.optimizationData);

    // function to handel the dependencies and call the SubMulOperator
    auto unwrapper = [&x]
            (SubVectorValues_future subYv_f,
             SubMatrix_future       subMtx_f,
             std::vector<SubVectorValues_future> subXv_fs
            )
            {
                return subMul( subMtx_f.get(), x.values, subYv_f.get() );
            };

    // loop over all subdomains
    for(size_t i=0; i<subYs.size(); ++i)
    {
        SubVector & subY = subYs.[i];
        SubVector & subX = subXs.[i];
        SubDomain & subA = subAs.[i];

        HPX_ASSERT(subX.subLength == subY.subLength);

        // Vector of all SubVector neighbors of this SubDomain from the x Vector
        // TODO dies auch async machen?
        std::vector<SubVectorValues_future> subXv_fs;
        subXv_fs.reserve(subA.dependencies.size() );
        for (std::set<int>::iterator it = subA.dependencies.begin();
                it != subA.dependencies.end(); ++it)
        {
            subXv_fs.push_back( subXs.[*it].subValues_f );
        }

        // make the async call
        using hpx::lcos::local::dataflow;
        subY.subValues_f = dataflow(
                hpx::launch::async, unwrapper,
                subY.subValues_f,
                subA.subMatrix_f,
                subXv_fs
            );

    }

    return 0;
}

// wrapper function
int ComputeSPMV(const  SparseMatrix & A, Vector & x, Vector & y) {
  // Test vector lengths
  assert(x.localLength>=A.localNumberOfColumns);
  assert(y.localLength>=A.localNumberOfRows);

  A.isSpmvOptimized = true;
  
  // version async
  /*return ComputeSPMV_async(A, x, y).wait(), 0;*/

  // Version async inc. sub domains
  return ComputeSPMV_sub_async(A, x, y);

}

#endif
