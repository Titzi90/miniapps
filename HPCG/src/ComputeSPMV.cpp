
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


//hpx::future<void> ComputeSPMV_async( const SparseMatrix & A, [>const<] Vector & x, Vector & y) {

  //assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  //assert(y.localLength>=A.localNumberOfRows);

//#ifndef HPCG_NOMPI
    //ExchangeHalo(A,x);
//#endif

  //const double * const xv = x.values;
  //double * const yv = y.values;
  //const local_int_t nrow = A.localNumberOfRows;

  //typedef boost::counting_iterator<local_int_t> iterator;

  //return hpx::parallel::for_each(
    //hpx::parallel::task, iterator(0), iterator(nrow),
    //[xv, yv, &A](local_int_t i) {
      //double sum = 0.0;
      //const double * const cur_vals = A.matrixValues[i];
      //const local_int_t * const cur_inds = A.mtxIndL[i];
      //const int cur_nnz = A.nonzerosInRow[i];

      //for (int j=0; j< cur_nnz; j++)
        //sum += cur_vals[j]*xv[cur_inds[j]];
      //yv[i] = sum;
    //});
//}

/******************************************************************************/

//TODO ref und const überprüfen

// Matrix Vector Mul of one row
inline double mul(double const * mtxVal,
                  local_int_t const * mtxInd,
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
// subYv = SubMtx * vecVal
inline std::vector<double*> subMul(MatrixValues const subMtx,
                                   double const * vecVal,
                                   std::vector<double*> subYv)
{
    for (local_int_t row=0; row<subYv.size(); ++row)
    {
        *subYv[row] = mul(subMtx.values[row],
                          subMtx.indLoc[row],
                          *subMtx.nonzerosInRow[row],
                          vecVal);
    }
    return subYv;
}


// wrapper arund mul function for async call
// waits untill all dependencis are avilible and then calls mul function
hpx::future<std::vector<double*> > asyncMul(MatrixValues_furure& mtx_f,
                                            SubVector& vec,
                                            VectorValues_future& res_f)
{
    //helperfuntion unwraping wait construct and calling mul function
    //hpx::util::bind(<func>, <parameterliste inc. placeholder>)
    auto unwrapper =
        hpx::util::bind([](double const * vecVal,
                           hpx::future<hpx::util::tuple<
                               MatrixValues_furure,
                               hpx::future<std::vector<VectorValues_future> >,
                               VectorValues_future
                           > > arg_f)
                          -> std::vector<double*>
                          {
                            auto arg = arg_f.get();
                            return subMul(hpx::util::get<0>(arg).get(),
                                          vecVal,
                                          hpx::util::get<2>(arg).get());
                          },
                        vec.localetyValues,
                        hpx::util::placeholders::_1
                       );

    //wait for all dependencis and then call the unwrapper
    return hpx::when_all(mtx_f,
                         hpx::when_all(vec.getNeighbourhood()),
                         res_f
                        ).then(hpx::launch::async, unwrapper);
}


int ComputeSPMV_sub_async(SparseMatrix const & A, Vector  & x, Vector& y )
{
#ifdef HPCG_DEBUG
    std::cout << "using SPMV_sub_async" << std::endl;
#endif
//TODO more localytsy
//TODO zusammenschieben?

  std::vector<SubDomain>  & subAs =
      *static_cast<std::vector<SubDomain>* >(A.optimizationData);
  std::vector<SubVector>  & subXs =
      *static_cast<std::vector<SubVector>* >(x.optimizationData);
  std::vector<SubVector>& subYs =
      *static_cast<std::vector<SubVector>* >(y.optimizationData);

  // loop over all subdomains
    for(size_t i=0; i<subYs.size(); ++i)
    {
        SubVector  & subY = subYs.at(i);
        SubVector  & subX = subXs.at(i);
        SubDomain  & subA = subAs.at(i);

        HPX_ASSERT(subX.localLength == subY.localLength);

        subY.values_f = asyncMul(subA.matrixValues,
                                 subX,
                                 subY.values_f);
    }

  return 0;
}






int ComputeSPMV(const  SparseMatrix & A, Vector & x, Vector & y) {
  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

  A.isSpmvOptimized = true;
  
  // version async
  //return ComputeSPMV_async(A, x, y).wait(), 0;

  //Version async inc. sub domains
  return ComputeSPMV_sub_async(A, x, y);

}

#endif
