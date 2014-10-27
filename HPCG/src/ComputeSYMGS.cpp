
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#if !defined(HPCG_NOHPX)
#include <hpx/hpx_fwd.hpp>
#endif

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

/*!
  Routine to one step of symmetrix Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in]  A the known system matrix
  @param[in]  x the input vector
  @param[out] y On exit contains the result of one symmetric GS sweep with x as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSYMGS_ref
*/

#ifdef HPCG_NOHPX
int ComputeSYMGS( const SparseMatrix & A, const Vector & x, Vector & y) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  return(ComputeSYMGS_ref(A, x, y));

}

#else
// HPX version
#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/lcos/when_all.hpp>
#include <algorithm>
#include "OptimizeProblem.hpp"

//TODO alternativ forward backward seep über bei 

/**
 * performing the Sym. Gaus-Seidel step inside one Sub Domain
 */
//TODO warum kann ich subY nicht als ref übergben?
std::vector<double*> subSYMGS(MatrixValues const & subMtx,
                                     std::vector<double*> const & subRHS,
                                     std::vector<double*>  subY,
                                     double const * yValus)
{
    //forward sweep
    for (local_int_t i=0; i<subY.size(); ++i){
        double sum = * (subRHS.at(i) );
        for (char j=0; j<*( subMtx.nonzerosInRow.at(i) ); ++j){
            sum -= subMtx.values.at(i)[j] * yValus[subMtx.indLoc.at(i)[j] ];
        }
        //add diagonal value again as we subtracted it in the uper loop
        sum += *( subMtx.diagonal.at(i) ) * (*(subY.at(i)));

        // divide by diagonal value and subscibe it to y-vector
        *subY.at(i) = sum / (*subMtx.diagonal.at(i) );
    }
    //back sweep
    for (local_int_t i=subY.size()-1; i>=0; --i){
        double sum = * (subRHS.at(i) );
        for (char j=0; j<*( subMtx.nonzerosInRow.at(i) ); ++j){
            sum -= subMtx.values.at(i)[j] * yValus[subMtx.indLoc.at(i)[j] ];
        }
        //add diagonal value again as we subtracted it in the uper loop
        sum += *( subMtx.diagonal.at(i) ) * (*(subY.at(i)));

        // divide by diagonal value and subscibe it to y-vector
        *(subY.at(i)) = sum / (*(subMtx.diagonal.at(i)) );
    }
    return subY;
}

/**
 * This version of the symetic Gaus Siedel solver make use of the sub domains
 * and using HPX to work on each sub Domain asyncron
 */
int ComputeSYMGS_subDom(SparseMatrix const & A, Vector const & rhs , Vector & y){

#ifdef HPCG_DEBUG
    std::cout << "using SYMGS_subDom" << std::endl;
#endif
    //TODO more localytsy

    std::vector<SubDomain>  & subAs =
        *static_cast<std::vector<SubDomain>* >(A.optimizationData);
    std::vector<SubVector>  & subRHSs =
        *static_cast<std::vector<SubVector>* >(rhs.optimizationData);
    std::vector<SubVector>& subYs =
        *static_cast<std::vector<SubVector>* >(y.optimizationData);

    //unwrapper functon calls subSYMGS if all dependencis are ready
    //TODO check ob ich beim async call y zu welchen ZEitpunkt kopiere?
    //     wenn problem dann bind
    auto unwrapper =[y](MatrixValues_furure mtx_f,
                        VectorValues_future rhs_f,
                        VectorValues_future y_f,
                        std::vector<VectorValues_future>  neighbourhood
                    )-> std::vector<double*>
                    {
                        return subSYMGS(mtx_f.get(),
                                        rhs_f.get(),
                                        y_f.get(),
                                        y.values
                                       );
                     };

    // loop over all subdomains
    // nicht (!) parralel
    for(size_t i=0; i<subYs.size(); ++i)
    {
        SubVector & subY   = subYs.at(i);
        SubVector & subRHS = subRHSs.at(i);
        SubDomain & subA   = subAs.at(i);

        HPX_ASSERT(subRHS.localLength == subY.localLength);

        subY.values_f = hpx::lcos::local::dataflow(
                            hpx::launch::async,
                            unwrapper,
                            subA.matrixValues,
                            subRHS.values_f,
                            subY.values_f,
                            subY.getNeighbourhood()
                        );
    }
}

int ComputeSYMGS( const SparseMatrix & A, const Vector & x, Vector & y) {
  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

  A.isSpmvOptimized = true;
  
  return ComputeSYMGS_subDom(A, x, y);

}
#endif
