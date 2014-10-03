
//@HEADER
// ***************************************************
//
// HPCG: testing routin
//
//
// ***************************************************
//@HEADER


#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cassert>

#ifndef HPCG_NOHPX
#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/performance_counters/performance_counter.hpp>
#endif

#include "hpcg.hpp"
#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "CG.hpp"
#include "CG_ref.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"

#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;



int main(int argc, char * argv[]) {


#ifndef HPCG_NOHPX
if(0 == hpx::get_locality_id())
{
  hpx::cerr<<"Using HPX"<<hpx::endl;
}
#endif
#ifndef HPCG_NOOPENMP
  std::cerr<<"Using OMP"<<std::endl;
#endif
#ifndef HPCG_NOMPI
  std::cerr<<"Using MPI"<<std::endl;
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes/HPX localitys, My MPI process ID/HPX locality ID

#ifdef HPCG_DETAILED_DEBUG
#ifndef HPCG_NOHPX
  HPCG_fout << "Process "<<rank<<" of "<<size<<" is alive with " << params.numThreads << " threads.\n" << hpx::flush;
#endif

  /*
  if (rank==0) {
    char c;
    std::cout << "Press key to continue"<< std::endl;
    std::cin.get(c);
  }
  */
#endif

  local_int_t nx,ny,nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;  // Used to check return codes on function calls

  // //////////////////////
  // Problem setup Phase //
  /////////////////////////

// setup Problem as usualy

#ifdef HPCG_DEBUG
  double t1 = mytimer();
#endif

  // Construct the geometry and linear system
  Geometry * geom_ref = new Geometry;
  Geometry * geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, nx, ny, nz, geom_ref);
  GenerateGeometry(size, rank, params.numThreads, nx, ny, nz, geom);

  SparseMatrix A_ref;
  SparseMatrix A;
  InitializeSparseMatrix(A_ref, geom_ref);
  InitializeSparseMatrix(A, geom);

  Vector b_ref, x_ref, xexact_ref;
  Vector b, x, xexact;
  GenerateProblem(A_ref, &b_ref, &x_ref, &xexact_ref);
  GenerateProblem(A, &b, &x, &xexact);
  SetupHalo(A_ref);
  SetupHalo(A);
  int numberOfMgLevels = 4; // Number of levels including first
  SparseMatrix * curLevelMatrix_ref = &A_ref;
  for (int level = 1; level< numberOfMgLevels; ++level) {
      GenerateCoarseProblem(*curLevelMatrix_ref);
      curLevelMatrix_ref = curLevelMatrix_ref->Ac; // Make the just-constructed coarse grid the next level
  }
  SparseMatrix * curLevelMatrix = &A;
  for (int level = 1; level< numberOfMgLevels; ++level) {
      GenerateCoarseProblem(*curLevelMatrix);
      curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
  }


  CGData data_ref;
  CGData data;
  InitializeSparseCGData(A_ref, data_ref);
  InitializeSparseCGData(A, data);


  // Use this array for collecting timing information
  std::vector< double > times(9,0.0);



  // Call user-tunable set up function.
  double t7 = mytimer(); OptimizeProblem(A, data, b, x, xexact); t7 = mytimer() - t7;
  times[7] = t7;

#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
  if (rank==0) std::cerr << "Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
  if (rank==0) HPCG_fout << "Optimization setuptime (sec) = " << times[7] << endl;
  if (rank==0) std::cerr << "Optimization setuptime (sec) = " << times[7] << endl;
#endif


/******************************************************************************/
#ifndef HPCG_NOHPX
  hpx::cerr << "we have " << size << " localitys with " 
            << params.numThreads << " hpx threads" <<hpx::endl;


#ifdef HPCG_DEBUG
  hpx::cerr << "startin init test..." << hpx::endl;
// compare data (ohne MG data)
  HPX_ASSERT(b_ref.localLength == b.localLength);
  HPX_ASSERT(x_ref.localLength == x.localLength);
  HPX_ASSERT(data_ref.r.localLength == data.r.localLength);
  HPX_ASSERT(data_ref.z.localLength == data.z.localLength);
  HPX_ASSERT(data_ref.p.localLength == data.p.localLength);
  HPX_ASSERT(data_ref.Ap.localLength == data.Ap.localLength);
  HPX_ASSERT(A_ref.totalNumberOfRows == A.totalNumberOfRows);
  HPX_ASSERT(A_ref.totalNumberOfNonzeros == A.totalNumberOfNonzeros);
  HPX_ASSERT(A_ref.localNumberOfRows == A.localNumberOfRows);
  HPX_ASSERT(A_ref.localNumberOfColumns == A.localNumberOfColumns);
  HPX_ASSERT(A_ref.localNumberOfNonzeros == A.localNumberOfNonzeros);
  for (local_int_t i=0; i<b.localLength; ++i){
      HPX_ASSERT(b_ref.values[i] == b.values[i]);
  }
  for (local_int_t i=0; i<x.localLength; ++i){
      HPX_ASSERT(x_ref.values[i] == x.values[i]);
  }
  for (local_int_t i=0; i<A.localNumberOfRows; ++i){
      HPX_ASSERT(A_ref.nonzerosInRow[i] == A.nonzerosInRow[i]);
      HPX_ASSERT(*A_ref.matrixDiagonal[i] == *A.matrixDiagonal[i]);
    for (int j=0; j<A.nonzerosInRow[i]; ++j){
        HPX_ASSERT(A_ref.matrixValues[i][j] == A.matrixValues[i][j]);
        HPX_ASSERT(A_ref.mtxIndG[i][j] == A.mtxIndG[i][j]);
        HPX_ASSERT(A_ref.mtxIndL[i][j] == A.mtxIndL[i][j]);
    }
  }
  hpx::cerr << "init test ok!" <<hpx::endl;
#endif

  std::vector<SubVector>  & subBs =
        *static_cast<std::vector<SubVector>* >(x.optimizationData);
hpx::cerr << "We have " << subBs.size() << " sub domains" << hpx::endl;
#ifdef HPCG_DEBUG
  for(size_t i=0; i<subBs.size(); ++i){
      VectorValues_future values_f = subBs[i].values_f;
      std::vector<double*> values = values_f.get();
      HPX_ASSERT(values.size() == subBs[i].localLength);
  }
#endif

#endif

/******************************************************************************/
// SPMV TEST
  std::cerr << "starting SPMV test" <<std::endl;
  //reverence
  double ref_time = mytimer();
  ComputeSPMV_ref(A_ref, b_ref, x_ref);
  ref_time = mytimer() - ref_time;

#ifndef HPCG_NOHPX
  //define and reset counter
  hpx::performance_counters::performance_counter idle_rate (
          "/threads{localiti#0/total}/idle-rate");
  hpx::performance_counters::performance_counter thread_counter (
          "/threads{localiti#0/total}/count/cumulative");
  idle_rate.reset_sync();
  thread_counter.reset_sync();
  double opt_time = mytimer();
  ComputeSPMV    (A   , b     , x    );
  hpx::wait_all(when_vec(x));
  opt_time = mytimer() - opt_time;
  hpx::cout << "idle rate: " << 0.01 * idle_rate.get_value_sync<int>() << "%"<< hpx::endl;
  hpx::cout << thread_counter.get_value_sync<int>() << " number of hpx threads"<< hpx::endl;
  
#ifdef HPCG_DEBUG
  //varivate data
  for (local_int_t i=0; i<x.localLength; ++i){
    HPX_ASSERT(x_ref.values[i] == x.values[i]);
  }
  hpx::cerr << "SPMV test ok!\n";
#endif
  hpx::cout << "ref_time was " <<ref_time << " opt_time was " << opt_time <<hpx::endl;

#else

  std::cout << "SPMV toks " <<ref_time << std::endl;

#endif



#ifdef HPCG_DETAILED_DEBUG
  //if (geom->size == 1) WriteProblem(*geom, A, b, x, xexact);
#endif






  // Clean up
  DeleteMatrix(A_ref); // This delete will recursively delete all coarse grid data
  DeleteMatrix(A); // This delete will recursively delete all coarse grid data
  DeleteCGData(data_ref);
  DeleteCGData(data);
  DeleteVector(x_ref);
  DeleteVector(x);
  DeleteVector(b_ref);
  DeleteVector(b);
  DeleteVector(xexact_ref);
  DeleteVector(xexact);

  HPCG_Finalize();

  // Finish up

#ifdef HPCG_DEBUG
  HPCG_fout << "Total report results phase execution time in main (sec) = " << mytimer() - t1 << endl;
#endif
  return 0 ;
}
