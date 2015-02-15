
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
#include <hpx/include/iostreams.hpp>

#include <algorithm>
#include "OptimizeProblem.hpp"

/**********ONE STEP VERSION****************************************************/

// Performe one Gauss Seidel stepp
// ele is the element in the subY to be calculate
inline void GSStepper(SubMatrix const & subMtx,
                      SubVectorValues const & subRHS,
                      double const * yValus,
                      SubVectorValues & subY,
                      size_t const ele
                     )
{
    double sum = *subRHS.at(ele);

    for (char j = 0; j < *subMtx.nonzerosInRow.at(ele); ++j)
    {
        sum -= subMtx.values.at(ele)[j] * yValus[ subMtx.indLoc.at(ele)[j] ];
    }

    // add diagonal value again as we erroneous subtracted it in the uper loop
    sum += ( *subMtx.diagonal.at(ele) ) * ( *subY.at(ele) );

    // divide by diagonal value and subscibe it to y-vector
    *subY.at(ele) = sum / ( *subMtx.diagonal.at(ele) );
}

/**
 * performing the Sym. Gaus-Seidel step inside one Sub Domain
 */
// TODO forward back swap auf lacelety machen nicht in subdomains machen
//      -> Vergleichen
SubVectorValues subSYMGS(SubMatrix const & subMtx,
                           SubVectorValues const & subRHS,
                           double const * yValus,
                           SubVectorValues  subY    //TODO warum keine Referenz möglich?
                          )
{
    //forward sweep
    for (size_t i=0; i<subY.size(); ++i)
    {
        GSStepper(subMtx, subRHS, yValus, subY, i);
    }
    //back sweep
    for (local_int_t i=subY.size()-1; i>=0; --i)
    {
        GSStepper(subMtx, subRHS, yValus, subY, i);
    }

    return subY;
}

/**
 * This version of the symetic Gaus Siedel solver make use of the sub domains
 * and using HPX to work on each sub Domain asyncron
 * Moreover it computes the secound Symetic Gaus Seidel step 
 * just inside each SubDomain
 */
int ComputeSYMGS_sub_async(SparseMatrix const & A,
                           Vector const & rhs ,
                           Vector & y
                          )
{

#ifdef HPCG_DEBUG
    std::cout << "using SYMGS_sub_async" << std::endl;
#endif
    //TODO more localytsy

    std::vector<SubDomain>  & subAs =
        *static_cast<std::vector<SubDomain>* >(A.optimizationData);
    std::vector<SubVector>  & subRHSs =
        *static_cast<std::vector<SubVector>* >(rhs.optimizationData);
    std::vector<SubVector>& subYs =
        *static_cast<std::vector<SubVector>* >(y.optimizationData);

    // loop over all subdomains
    for(size_t i=0; i<subYs.size(); ++i)
    {
        SubVector & subY   = subYs.at(i);
        SubVector & subRHS = subRHSs.at(i);
        SubDomain & subA   = subAs.at(i);

        HPX_ASSERT(subRHS.subLength == subY.subLength);

        // Vector of all SubVector neighbors of this SubDomain from the x Vector
        // TODO dies auch async machen?
        std::vector<SubVectorValues_future> subYv_fs;
        subYv_fs.reserve(subA.dependencies.size() );
        for (std::set<int>::iterator it = subA.dependencies.begin();
                it != subA.dependencies.end(); ++it)
        {
            subYv_fs.push_back( subYs.at(*it).subValues_f );
        }

        // function to handel and unwrapp the dependencis
        // and call subSYMGS
        auto unwrapper = [&y, subY]
                (SubMatrix_future       subMtx_f,
                 SubVectorValues_future subRHSv_f,
                 std::vector<SubVectorValues_future> subYv_fs
                )
                {
                    return subSYMGS(subMtx_f.get(),
                                    subRHSv_f.get(),
                                    y.values,
                                    subY.subValues_f.get()
                                   );
                };

        // make the async call
        using hpx::lcos::local::dataflow;
        subY.subValues_f = dataflow(
                hpx::launch::async, unwrapper,
                subA.subMatrix_f,
                subRHS.subValues_f,
                subYv_fs
            );
    }

    return 0;
}

/**********TWO STEP VERSION****************************************************/

//forward sweep
SubVectorValues subSYMGS_forward(SubMatrix const & subMtx,
                                 SubVectorValues const & subRHS,
                                 double const * yValus,
                                 SubVectorValues  subY    //TODO warum keine Referenz möglich?
                                )
{
    for (size_t i=0; i<subY.size(); ++i)
    {
        GSStepper(subMtx, subRHS, yValus, subY, i);
    }
    return subY;
}

//back sweep
SubVectorValues subSYMGS_revers(SubMatrix const & subMtx,
                                SubVectorValues const & subRHS,
                                double const * yValus,
                                SubVectorValues  subY    //TODO warum keine Referenz möglich?
                               )
{
    for (local_int_t i=subY.size()-1; i>=0; --i)
    {
        GSStepper(subMtx, subRHS, yValus, subY, i);
    }

    return subY;
}


/**
 * This version of the symetic Gaus Siedel solver make use of the sub domains
 * and using HPX to work on each sub Domain asyncron
 * Morover it computes the second step of the Symetic GausSeidel after
 * all depending Subdomains are finished with first step
 * and calls the secound sweap in same order
 */
int ComputeSYMGS_sub_async_twostep(SparseMatrix const & A,
                                   Vector const & rhs ,
                                   Vector & y
                                  )
{

#ifdef HPCG_DEBUG
    std::cout << "using SYMGS_sub_async_twosetp" << std::endl;
#endif
    //TODO more localytsy

    std::vector<SubDomain>  & subAs =
        *static_cast<std::vector<SubDomain>* >(A.optimizationData);
    std::vector<SubVector>  & subRHSs =
        *static_cast<std::vector<SubVector>* >(rhs.optimizationData);
    std::vector<SubVector>& subYs =
        *static_cast<std::vector<SubVector>* >(y.optimizationData);

    // first loop over all subdomains
    for(size_t i=0; i<subYs.size(); ++i)
    {
        SubVector & subY   = subYs.at(i);
        SubVector & subRHS = subRHSs.at(i);
        SubDomain & subA   = subAs.at(i);

        HPX_ASSERT(subRHS.subLength == subY.subLength);

        // Vector of all SubVector neighbors of this SubDomain from the x Vector
        std::vector<SubVectorValues_future> subYv_fs;
        subYv_fs.reserve(subA.dependencies.size() );
        for (std::set<int>::iterator it = subA.dependencies.begin();
                it != subA.dependencies.end(); ++it)
        {
            subYv_fs.push_back( subYs.at(*it).subValues_f );
        }

        // function to handel and unwrapp the dependencis
        // and call subSYMGS
        auto unwrapper = [&y, subY]
                (SubMatrix_future       subMtx_f,
                 SubVectorValues_future subRHSv_f,
                 std::vector<SubVectorValues_future> subYv_fs
                )
                {
                    return subSYMGS_forward(subMtx_f.get(),
                                            subRHSv_f.get(),
                                            y.values,
                                            subY.subValues_f.get()
                                           );
                };

        // make the async call
        using hpx::lcos::local::dataflow;
        subY.subValues_f = dataflow(
                hpx::launch::async, unwrapper,
                subA.subMatrix_f,
                subRHS.subValues_f,
                subYv_fs
            );
    }

    // secound loop over all subdomains
    for(size_t i=0; i<subYs.size(); ++i)
    {
        SubVector & subY   = subYs.at(i);
        SubVector & subRHS = subRHSs.at(i);
        SubDomain & subA   = subAs.at(i);

        HPX_ASSERT(subRHS.subLength == subY.subLength);

        // Vector of all SubVector neighbors of this SubDomain from the x Vector
        std::vector<SubVectorValues_future> subYv_fs;
        subYv_fs.reserve(subA.dependencies.size() );
        for (std::set<int>::iterator it = subA.dependencies.begin();
                it != subA.dependencies.end(); ++it)
        {
            subYv_fs.push_back( subYs.at(*it).subValues_f );
        }

        // function to handel and unwrapp the dependencis
        // and call subSYMGS
        auto unwrapper = [&y, subY]
                (SubMatrix_future       subMtx_f,
                 SubVectorValues_future subRHSv_f,
                 std::vector<SubVectorValues_future> subYv_fs
                )
                {
                    return subSYMGS_revers(subMtx_f.get(),
                                           subRHSv_f.get(),
                                           y.values,
                                           subY.subValues_f.get()
                                          );
                };

        // make the async call
        using hpx::lcos::local::dataflow;
        subY.subValues_f = dataflow(
                hpx::launch::async, unwrapper,
                subA.subMatrix_f,
                subRHS.subValues_f,
                subYv_fs
            );
    }

    return 0;
}

/**********TWO STEP REVERSE VERSION********************************************/

/**
 * This version of the symetic Gaus Siedel solver make use of the sub domains
 * and using HPX to work on each sub Domain asyncron
 * Morover it computes the second step of the Symetic GausSeidel after
 * all depending Subdomains are finished with first step
 * and calls the secound seap in revers order
 */
int ComputeSYMGS_sub_async_twostep_revers(SparseMatrix const & A,
                                          Vector const & rhs ,
                                          Vector & y
                                         )
{

#ifdef HPCG_DEBUG
    std::cout << "using SYMGS_sub_async_twosetp_reverse" << std::endl;
#endif
    //TODO more localytsy

    std::vector<SubDomain>  & subAs =
        *static_cast<std::vector<SubDomain>* >(A.optimizationData);
    std::vector<SubVector>  & subRHSs =
        *static_cast<std::vector<SubVector>* >(rhs.optimizationData);
    std::vector<SubVector>& subYs =
        *static_cast<std::vector<SubVector>* >(y.optimizationData);

    // first loop over all subdomains
    for(size_t i=0; i<subYs.size(); ++i)
    {
        SubVector & subY   = subYs.at(i);
        SubVector & subRHS = subRHSs.at(i);
        SubDomain & subA   = subAs.at(i);

        HPX_ASSERT(subRHS.subLength == subY.subLength);

        // Vector of all SubVector neighbors of this SubDomain from the x Vector
        std::vector<SubVectorValues_future> subYv_fs;
        subYv_fs.reserve(subA.dependencies.size() );
        for (std::set<int>::iterator it = subA.dependencies.begin();
                it != subA.dependencies.end(); ++it)
        {
            subYv_fs.push_back( subYs.at(*it).subValues_f );
        }

        // function to handel and unwrapp the dependencis
        // and call subSYMGS
        auto unwrapper = [&y, subY]
                (SubMatrix_future       subMtx_f,
                 SubVectorValues_future subRHSv_f,
                 std::vector<SubVectorValues_future> subYv_fs
                )
                {
                    return subSYMGS_forward(subMtx_f.get(),
                                            subRHSv_f.get(),
                                            y.values,
                                            subY.subValues_f.get()
                                           );
                };

        // make the async call
        using hpx::lcos::local::dataflow;
        subY.subValues_f = dataflow(
                hpx::launch::async, unwrapper,
                subA.subMatrix_f,
                subRHS.subValues_f,
                subYv_fs
            );
    }

    // secound loop over all subdomains
    for(int i=subYs.size()-1; i>=0; --i)
    {
        SubVector & subY   = subYs.at(i);
        SubVector & subRHS = subRHSs.at(i);
        SubDomain & subA   = subAs.at(i);

        HPX_ASSERT(subRHS.subLength == subY.subLength);

        // Vector of all SubVector neighbors of this SubDomain from the x Vector
        std::vector<SubVectorValues_future> subYv_fs;
        subYv_fs.reserve(subA.dependencies.size() );
        for (std::set<int>::iterator it = subA.dependencies.begin();
                it != subA.dependencies.end(); ++it)
        {
            subYv_fs.push_back( subYs.at(*it).subValues_f );
        }

        // function to handel and unwrapp the dependencis
        // and call subSYMGS
        auto unwrapper = [&y, subY]
                (SubMatrix_future       subMtx_f,
                 SubVectorValues_future subRHSv_f,
                 std::vector<SubVectorValues_future> subYv_fs
                )
                {
                    return subSYMGS_revers(subMtx_f.get(),
                                      subRHSv_f.get(),
                                      y.values,
                                      subY.subValues_f.get()
                                     );
                };

        // make the async call
        using hpx::lcos::local::dataflow;
        subY.subValues_f = dataflow(
                hpx::launch::async, unwrapper,
                subA.subMatrix_f,
                subRHS.subValues_f,
                subYv_fs
            );
    }

    return 0;
}


/**********WRAPPER FUNCTION****************************************************/
int ComputeSYMGS( const SparseMatrix & A, const Vector & x, Vector & y) {
  // Test vector lengths
  assert(x.localLength>=A.localNumberOfColumns);
  assert(y.localLength>=A.localNumberOfRows);

  A.isSpmvOptimized = true;
  
  //return ComputeSYMGS_sub_async        (A, x, y);
  return ComputeSYMGS_sub_async_twostep(A, x, y);
  //return ComputeSYMGS_sub_async_twostep_revers(A, x, y);

}
#endif
