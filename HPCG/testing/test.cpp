//:
//@HEADER
// ***************************************************
//
// HPCG: testing routin
//
//
// ***************************************************
//@HEADER

// INCLUDES
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
#include "ComputeSYMGS_ref.hpp"
#include "ComputeSYMGS.hpp"
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


//#ifndef HPCG_NOHPX
//#define COUT hpx::cout
//#define CERR hpx::cerr
//#define ENDL hpx::endl
//#else
#define COUT std::cout
#define CERR std::cerr
#define ENDL std::endl
//#endif


#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif

int main(int argc, char * argv[]) {

CERR.precision(5);
COUT.precision(5);

#ifndef HPCG_NOHPX
if(0 == hpx::get_locality_id())
{
  CERR<<"Using HPX"<<ENDL;
}
#endif
#ifndef HPCG_NOOPENMP
  CERR<<"Using OMP"<<ENDL;
#endif
#ifndef HPCG_NOMPI
  CERR<<"Using MPI"<<ENDL;
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  // Number of MPI processes/HPX localitys, My MPI process ID/HPX locality ID
  int size = params.comm_size;
  int rank = params.comm_rank;

#ifdef HPCG_DETAILED_DEBUG
#ifndef HPCG_NOHPX
  HPCG_fout << "Process "<<rank<<" of "<<size<<" is alive with "
            << params.numThreads << " threads.\n" << hpx::flush;
#endif

  /*
  if (rank==0) {
    char c;
    COUT << "Press key to continue"<< ENDL;
    std::cin.get(c);
  }
  */
#endif

  local_int_t nx,ny,nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;  // Used to check return codes on function calls

/***********PROBLEM SETUP PHASE***********************************************/

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
  /*    TODO MG
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
  */

  CGData data_ref;
  CGData data;
  InitializeSparseCGData(A_ref, data_ref);
  InitializeSparseCGData(A, data);

  // Use this array for collecting timing information
  std::vector< double > times(9,0.0);

  // Call user-tunable set up function.
  double t7 = mytimer(); OptimizeProblem(A, data, b, x, xexact);
  t7 = mytimer() - t7;
  times[7] = t7;

#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total problem setup time in main (sec) = " << mytimer() - t1 << ENDL;
  if (rank==0) CERR << "Total problem setup time in main (sec) = " << mytimer() - t1 << ENDL;
  if (rank==0) HPCG_fout << "Optimization setuptime (sec) = " << times[7] << ENDL;
  if (rank==0) CERR << "Optimization setuptime (sec) = " << times[7] << ENDL;
#endif

{/***********CHECK FOR ERRORS WHILE SETUP**************************************/

#ifndef HPCG_NOHPX
  CERR << "we have " << size << " localitys with " 
            << params.numThreads << " hpx threads" <<ENDL;

#ifdef HPCG_DEBUG
  CERR << "startin init test..." << ENDL;
// compare data (ohne MG data)
  HPX_ASSERT(b_ref.localLength           == b.localLength);
  HPX_ASSERT(x_ref.localLength           == x.localLength);
  HPX_ASSERT(data_ref.r.localLength      == data.r.localLength);
  HPX_ASSERT(data_ref.z.localLength      == data.z.localLength);
  HPX_ASSERT(data_ref.p.localLength      == data.p.localLength);
  HPX_ASSERT(data_ref.Ap.localLength     == data.Ap.localLength);
  HPX_ASSERT(A_ref.totalNumberOfRows     == A.totalNumberOfRows);
  HPX_ASSERT(A_ref.totalNumberOfNonzeros == A.totalNumberOfNonzeros);
  HPX_ASSERT(A_ref.localNumberOfRows     == A.localNumberOfRows);
  HPX_ASSERT(A_ref.localNumberOfColumns  == A.localNumberOfColumns);
  HPX_ASSERT(A_ref.localNumberOfNonzeros == A.localNumberOfNonzeros);
  for (local_int_t i=0; i<b.localLength; ++i){
      HPX_ASSERT(b_ref.values[i] == b.values[i]);
  }
  for (local_int_t i=0; i<x.localLength; ++i){
      HPX_ASSERT(x_ref.values[i] == x.values[i]);
  }
  for (local_int_t i=0; i<A.localNumberOfRows; ++i){
      HPX_ASSERT( A_ref.nonzerosInRow[i]  ==  A.nonzerosInRow[i]);
      HPX_ASSERT(*A_ref.matrixDiagonal[i] == *A.matrixDiagonal[i]);
    for (int j=0; j<A.nonzerosInRow[i]; ++j){
        HPX_ASSERT(A_ref.matrixValues[i][j] == A.matrixValues[i][j]);
        HPX_ASSERT(A_ref.mtxIndG[i][j]      == A.mtxIndG[i][j]);
        HPX_ASSERT(A_ref.mtxIndL[i][j]      == A.mtxIndL[i][j]);
    }
  }
  std::vector<SubVector>  & subXs =
        *static_cast< std::vector<SubVector>* >(x.optimizationData);
  for(size_t i=0; i<subXs.size(); ++i){
      SubVectorValues_future values_f = subXs[i].subValues_f;
      SubVectorValues        values   = values_f.get();
      HPX_ASSERT(values.size() == subXs[i].subLength);
  }

  CERR << "init test ok!" <<ENDL;
#endif

  std::vector<SubVector>  & subBs =
        *static_cast< std::vector<SubVector>* >(b.optimizationData);
CERR << "We have " << subBs.size() << " sub domains" << ENDL;

#endif
}

{/************SPMV TEST*********************************************************/

#ifdef HPCG_DEBUG
#ifndef HPCG_NOHPX
  CERR << "starting SPMV test..." << ENDL;
  // reverence
  ComputeSPMV_ref(A_ref, b_ref, x_ref);
  // optimiced
  ComputeSPMV    (A   , b     , x    );

  // wait to finish computation
  when_vec(x).get();

  //varivate data
//CERR << "Values: #ele   ref    opt" << ENDL;
for (local_int_t i=0; i<x.localLength; ++i){
//CERR << i << "    " << x_ref.values[i] << "     " << x.values[i] << ENDL;
//if(x_ref.values[i] != x.values[i]) CERR << "ERROR by " << i << ": " << x_ref.values[i] << " " << x.values[i] << ENDL;
    HPX_ASSERT(x_ref.values[i] == x.values[i]);
  }

  CERR << "SPMV test ok!" << ENDL;
#endif
#endif
}

{/************SYMGS TEST********************************************************/

#ifdef HPCG_DEBUG
#ifndef HPCG_NOHPX
  CERR << "starting SYMGS test..." << ENDL;
 
  // Test with ones vector = soulution
  FillVector(x,1);
  ComputeSYMGS(A,b,x);

  when_vec(x).get();
  // every thing should stay one
  for (local_int_t i=0; i<x.localLength; ++i){
    HPX_ASSERT(1 == x.values[i]);
  }

  // Test with zero vector (x and b)
  FillVector(b,0);
  FillVector(x,0);
  ComputeSYMGS(A,b,x);

  when_vec(x).get();
  // every thing should stay zero
  for (local_int_t i=0; i<x.localLength; ++i){
    HPX_ASSERT(0 == x.values[i]);
  }

  CERR << "SYMGS test ok!" << ENDL;

#endif
#endif
}

{/************SPMV BENCHMARK****************************************************/
#ifndef HPCG_DEBUG

// no dependencis
  CERR << "starting SPMV benchmarks" << ENDL;

  double const BENCHTIME = 5;   // the benchmark should run at least for this time
  int const REPEAT = 100;       // the benchmark should run at least for this number of sets
  double time_ref = 0;          // time the benchmark takes
  double time_opt = 0;          // time the benchmark takes
  double time_Xdep = 0;         // time the benchmark takes
  double time_Bdep = 0;         // time the benchmark takes
  int repeat_ref = REPEAT;      // how offent the benchmark runs
  int repeat_opt = REPEAT;      // how offent the benchmark runs
  int repeat_Xdep = REPEAT;     // how offent the benchmark runs
  int repeat_Bdep = REPEAT;     // how offent the benchmark runs

  // recerence
  for (; time_ref<BENCHTIME; repeat_ref*=2){
    time_ref = mytimer();
    for(int r=0; r<repeat_ref; ++r){
      ComputeSPMV_ref(A_ref, b_ref, x_ref);
    }
    time_ref = mytimer() - time_ref;
  }
  repeat_ref /= 2;
  time_ref /= repeat_ref;
  COUT << time_ref << " sec. average (saple size: " << repeat_ref
       << ") non optimized runtime" << ENDL;

  //optimized version
#ifndef HPCG_NOHPX
  // define hpx countesrs and timer
  // TODO warum gehn die nemmer?
  hpx::performance_counters::performance_counter idleRate_counter (
          "/threads{localiti#0/total}/idle-rate");
  hpx::performance_counters::performance_counter thread_counter (
          "/threads{localiti#0/total}/count/cumulative");
  int idleRate =0;
  int threads =0;
  double thread_time =0.;

  for (; time_opt<BENCHTIME; repeat_opt*=2){
    idleRate_counter.reset_sync();
    thread_counter.reset_sync();
    time_opt = mytimer();
    for(int r=0; r<repeat_opt; ++r){
      ComputeSPMV    (A   , b     , x    );
      when_vec(x).get();    // wait to finish all computation
    }
    time_opt = mytimer() - time_opt;
    idleRate = idleRate_counter.get_value_sync<int>();
    threads   = thread_counter.get_value_sync<int>();
  }
  repeat_opt /= 2;
  thread_time = 1000. * time_opt/threads;
  time_opt /= repeat_opt;
  COUT << time_opt << " sec. average (saple size: " << repeat_opt
       << ") optimized hpx runtime (inc. barriers)" << ENDL;
  COUT << 0.01 * idleRate << "% total idle rate"   << ENDL;
  COUT << threads << " total number of hpx threads" << ENDL;
  COUT << thread_time << " ms work per thread" << ENDL;
  
// with dependencis of the target vector

  for (; time_Bdep<BENCHTIME; repeat_Bdep*=2){
    idleRate_counter.reset_sync();
    thread_counter.reset_sync();
    time_Bdep = mytimer();
    for(int r=0; r<repeat_Bdep; ++r){
      ComputeSPMV    (A   , b     , x    );
    }
    when_vec(x).get();    // wait to finish all computation
    time_Bdep = mytimer() - time_Bdep;
    idleRate  = idleRate_counter.get_value_sync<int>();
    threads   = thread_counter.get_value_sync<int>();
  }
  repeat_Bdep /= 2;
  thread_time = 1000. * time_opt/threads;   //TODO falsce Zeit
  time_Bdep /= repeat_Bdep;
  COUT << time_Bdep << " sec. average (saple size: " << repeat_Bdep
       << ") optimized hpx runtime with dependencis to the source vector" << ENDL;
  COUT << 0.01 * idleRate << "% total idle rate"   << ENDL;
  COUT << threads << " total number of hpx threads" << ENDL;
  COUT << thread_time << " ms work per thread" << ENDL;
  
// with dependencis of the source vector
  for (; time_Xdep<BENCHTIME; repeat_Xdep*=2){
    idleRate_counter.reset_sync();
    thread_counter.reset_sync();
    time_Xdep = mytimer();
    for(int r=0; r<repeat_Xdep; r+=2){
      ComputeSPMV    (A   , b     , x    );
      ComputeSPMV    (A   , x     , b    );
    }
    when_vec(b).get();    // wait to finish all computation
    time_Xdep = mytimer() - time_Xdep;
    idleRate = idleRate_counter.get_value_sync<int>();
    threads   = thread_counter.get_value_sync<int>();
  }
  repeat_Xdep /= 2;
  thread_time = 1000. * time_opt/threads;
  time_Xdep /= repeat_Xdep;
  COUT << time_Xdep << " sec. average (saple size: " << repeat_Xdep
       << ") optimized hpx runtime with dependencis to target vector" << ENDL;
  COUT << 0.01 * idleRate << "% total idle rate"   << ENDL;
  COUT << threads << " total number of hpx threads" << ENDL;
  COUT << thread_time << " ms work per thread" << ENDL;
  
#endif
#endif
}

{/************SYMGS BENCHMARK***************************************************/
#ifndef HPCG_DEBUG

  CERR << "starting SYMGS benchmarks" << ENDL;

  double const BENCHTIME = 5;   // the benchmark should run at least for this time
  int const REPEAT = 100;       // the benchmark should run at least for this number of sets
  double time_ref = 0;          // time the benchmark takes
  double time_opt = 0;          // time the benchmark takes
  double time_Xdep = 0;         // time the benchmark takes
  int repeat_ref = REPEAT;      // how offent the benchmark runs
  int repeat_opt = REPEAT;      // how offent the benchmark runs
  int repeat_Xdep = REPEAT ;    // how offent the benchmark runs

  // recerence
  for (; time_ref<BENCHTIME; repeat_ref*=2){
    time_ref = mytimer();
    for(int r=0; r<repeat_ref; ++r){
      ComputeSYMGS_ref(A_ref, b_ref, x_ref);
    }
    time_ref = mytimer() - time_ref;
  }
  repeat_ref /= 2;
  time_ref /= repeat_ref;
  COUT << time_ref << " sec. average (saple size: " << repeat_ref
       << ") non optimized runtime" << ENDL;

//optimized version
#ifndef HPCG_NOHPX
  hpx::performance_counters::performance_counter idleRate_counter (
          "/threads{localiti#0/total}/idle-rate");
  hpx::performance_counters::performance_counter thread_counter (
          "/threads{localiti#0/total}/count/cumulative");
  int idleRate =0;
  int threads =0;
  double thread_time =0.;

  for (; time_opt<BENCHTIME; repeat_opt*=2){
    idleRate_counter.reset_sync();
    thread_counter.reset_sync();
    time_opt = mytimer();
    for(int r=0; r<repeat_opt; ++r){
      ComputeSYMGS(A   , b     , x    );
      when_vec(x).get();    // wait to finish all computation
    }
    time_opt = mytimer() - time_opt;
    idleRate = idleRate_counter.get_value_sync<int>();
    threads   = thread_counter.get_value_sync<int>();
  }
  repeat_opt /= 2;
  thread_time = 1000. * time_opt/threads;
  time_opt /= repeat_opt;
  COUT << time_opt << " sec. average (saple size: " << repeat_opt
       << ") optimized hpx runtime (inc. barriers)" << ENDL;
  COUT << 0.01 * idleRate << "% total idle rate"   << ENDL;
  COUT << threads << " total number of hpx threads" << ENDL;
  COUT << thread_time << " ms work per thread" << ENDL;

// with dependencis of the target vector
  for (; time_Xdep<BENCHTIME; repeat_Xdep*=2){
    //idleRate_counter.reset_sync();
    //thread_counter.reset_sync();
    time_Xdep = mytimer();
    for(int r=0; r<repeat_Xdep; ++r){
      ComputeSYMGS   (A   , b     , x    );
    }
    when_vec(x).get();    // wait to finish all computation
    time_Xdep = mytimer() - time_Xdep;
    idleRate = idleRate_counter.get_value_sync<int>();
    threads   = thread_counter.get_value_sync<int>();
  }
  repeat_Xdep /= 2;
  thread_time = 1000. * time_opt/threads;
  time_Xdep /= repeat_Xdep;
  COUT << time_Xdep << " sec. average (saple size: " << repeat_Xdep
       << ") optimized hpx runtime with dependencis to the target vector" << ENDL;
  COUT << 0.01 * idleRate << "% total idle rate"   << ENDL;
  COUT << threads << " total number of hpx threads" << ENDL;
  COUT << thread_time << " ms work per thread" << ENDL;
  
#endif

#endif
}

/***********FINALISE**********************************************************/
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
  HPCG_fout << "Total report results phase execution time in main (sec) = "
            << mytimer() - t1 << ENDL;
#endif

  return 0 ;
}
