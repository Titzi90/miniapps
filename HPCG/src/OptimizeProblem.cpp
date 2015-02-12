
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"
#include <iostream>
#include <utility>
#include <cstddef>
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/

int const NL = 16;//number of local points in each direction

int OptimizeProblem(SparseMatrix & A,
                    CGData & data,
                    Vector & b,
                    Vector & x,
                    Vector & xexact)
{

// This function can be used to completely transform any part of the data structures.
#ifndef HPCG_NOHPX

#ifdef HPCG_DEBUG
    std::cerr << "start opti problem" << std::endl;
#endif


/*************** extract and calculate geometry datas *************************/

// number of points in [x,y,z]-direction per locality
int const nx = A.geom->nx;
int const ny = A.geom->ny;
int const nz = A.geom->nz;

// calc local subdomain geometry
// we need at least 16 points in each direction for MG
// TODO make it more dynamic
assert(nx%NL == 0 && ny%NL == 0 && nz%NL == 0);
//  number of SubDomains in [x,y,z]-direction in the locality
int const numSubDomainsX = nx / NL;
int const numSubDomainsY = ny / NL;
int const numSubDomainsZ = nz / NL;
int const numSubDomains  = numSubDomainsX * numSubDomainsY * numSubDomainsZ;

// number of points in [x,y,z]-direction per SubDomain
int const numSubPointsX = NL;
int const numSubPointsY = NL;
int const numSubPointsZ = NL;
int const numSubPoints  = numSubPointsX*numSubPointsY*numSubPointsZ;

// create and allocate Sub Domain and SubVector
A.optimizationData       = new std::vector<SubDomain>();
b.optimizationData       = new std::vector<SubVector>();
x.optimizationData       = new std::vector<SubVector>();
data.r.optimizationData  = new std::vector<SubVector>();
data.z.optimizationData  = new std::vector<SubVector>();
data.p.optimizationData  = new std::vector<SubVector>();
data.Ap.optimizationData = new std::vector<SubVector>();

std::vector<SubDomain> & subDomains =
    *static_cast<std::vector<SubDomain>* >(A.optimizationData);
std::vector<SubVector> & subBs =
    *static_cast<std::vector<SubVector>* >(b.optimizationData);
std::vector<SubVector> & subXs =
    *static_cast<std::vector<SubVector>* >(x.optimizationData);
std::vector<SubVector> & subRs =
    *static_cast<std::vector<SubVector>* >(data.r.optimizationData);
std::vector<SubVector> & subZs =
    *static_cast<std::vector<SubVector>* >(data.z.optimizationData);
std::vector<SubVector> & subPs =
    *static_cast<std::vector<SubVector>* >(data.p.optimizationData);
std::vector<SubVector> & subAps =
    *static_cast<std::vector<SubVector>* >(data.Ap.optimizationData);

subDomains.reserve(numSubDomains);
subBs.reserve(numSubDomains);
subXs.reserve(numSubDomains);
subRs.reserve(numSubDomains);
subZs.reserve(numSubDomains);
subPs.reserve(numSubDomains);
subAps.reserve(numSubDomains);

/*
 * TODO MG not implemented yet -> doen't need this yet
 * TODO check if everything is correct (variabel names ...)
//create subDomains for MultiGrid
SparseMatrix* Acur = &A;
while (0 != Acur->Ac){
    Acur->Ac->optimizationData          = new std::vector<SubDomain>();
    Acur->mgData->optimizationData      = new std::vector<SubF2C>();
    Acur->mgData->rc->optimizationData  = new std::vector<SubVector>();
    Acur->mgData->xc->optimizationData  = new std::vector<SubVector>();
    Acur->mgData->Axf->optimizationData = new std::vector<SubVector>();

    static_cast<std::vector<SubDomain>* >(Acur->Ac->optimizationData)
        ->reserve(numSubDomains);
    static_cast<std::vector<SubF2C>* >(Acur->mgData->optimizationData)
        ->reserve(numSubDomains);
    static_cast<std::vector<SubVector>* >(Acur->mgData->rc->optimizationData)
        ->reserve(numSubDomains);
    static_cast<std::vector<SubVector>* >(Acur->mgData->xc->optimizationData)
        ->reserve(numSubDomains);
    static_cast<std::vector<SubVector>* >(Acur->mgData->Axf->optimizationData)
        ->reserve(numSubDomains);

    Acur = Acur->Ac;
}
*/

// initialize SubDomains and SubVectors
for (int subID=0; subID<numSubDomains; ++subID){
    SubDomain subDomain;

    //compute x y z rank of subDomain/-Vector (index i --> cordinate x,y,z)
    int & subIDx = subDomain.subIDx;
    int & subIDy = subDomain.subIDy;
    int & subIDz = subDomain.subIDz;

    subDomain.subID = subID;
    subIDz = subID / (numSubDomainsX*numSubDomainsY);
    subIDy = (subID-subIDz*numSubDomainsX*numSubDomainsY) / numSubDomainsX;
    subIDx = subID % numSubDomainsX;

    assert(subID == index(subIDx, subIDy, subIDz,
                          numSubDomainsX, numSubDomainsY, numSubDomainsZ) );

    // allocate data pointers for Matrix
    SubMatrix subMatrix;

    std::vector<double*> & subValues = subMatrix.values;
    std::vector<double*> & matrixDiagonal = subMatrix.diagonal;
    std::vector<global_int_t*> & mtxIndG = subMatrix.indG;
    std::vector<local_int_t*> & mtxIndLoc = subMatrix.indLoc;
    std::vector<char*> & nonzerosInRow = subMatrix.nonzerosInRow;

    subValues.reserve(numSubPoints);
    matrixDiagonal.reserve(numSubPoints);
    mtxIndG.reserve(numSubPoints);
    mtxIndLoc.reserve(numSubPoints);
    nonzerosInRow.reserve(numSubPoints);

    // allocate data pointers for vector
    std::vector<double*> subBv;
    std::vector<double*> subXv;
    std::vector<double*> subRv;
    std::vector<double*> subZv;
    std::vector<double*> subPv;
    std::vector<double*> subApv;

    subBv.reserve(numSubPoints);
    subXv.reserve(numSubPoints);
    subRv.reserve(numSubPoints);
    subZv.reserve(numSubPoints);
    subPv.reserve(numSubPoints);
    subApv.reserve(numSubPoints);

    // compute pointers from SubMatrixes/SubVectors to localety matrix/vectors
    for (local_int_t iSz=0; iSz<numSubPointsZ; iSz++) {
        local_int_t iLz = subIDz*numSubPointsZ+iSz;
        for (local_int_t iSy=0; iSy<numSubPointsY; iSy++) {
            local_int_t iLy = subIDy*numSubPointsY+iSy;
            for (local_int_t iSx=0; iSx<numSubPointsX; iSx++) {
                local_int_t iLx = subIDx*numSubPointsX+iSx;

                local_int_t currentSubRow = iSz*numSubPointsX*numSubPointsY
                                          + iSy*numSubPointsX
                                          + iSx;
                local_int_t currentLocaletyRow = iLz*nx*ny
                                               + iLy*nx
                                               + iLx;

                // set Matrix links to original data
                subValues.push_back(A.matrixValues[currentLocaletyRow]);
                matrixDiagonal.push_back(A.matrixDiagonal[currentLocaletyRow]);
                mtxIndG.push_back(A.mtxIndG[currentLocaletyRow]);
                mtxIndLoc.push_back(A.mtxIndL[currentLocaletyRow]);
                nonzerosInRow.push_back(&A.nonzerosInRow[currentLocaletyRow]);

                // set Vector links to original data
                subBv.push_back(&b.values[currentLocaletyRow]);
                subXv.push_back(&x.values[currentLocaletyRow]);
                subRv.push_back(&data.r.values[currentLocaletyRow]);
                subZv.push_back(&data.z.values[currentLocaletyRow]);
                subPv.push_back(&data.p.values[currentLocaletyRow]);
                subApv.push_back(&data.Ap.values[currentLocaletyRow]);

                // collecting all dependencies of the SubDomains
                // TODO save symetris
                for (int i=0; i!=*nonzerosInRow[currentSubRow]; ++i){
                    subDomain.dependencies.insert( belongsTo(
                                mtxIndG.at(currentSubRow)[i],
                                numSubDomainsX, numSubDomainsY, numSubDomainsZ,
                                numSubPointsX, numSubPointsY, numSubPointsZ
                                ) );
                }
            }
        }
    }

    // save gernerl geometry information to SubDomain (on uper level (A) )
    subDomain.numSubDomainsX = numSubDomainsX;
    subDomain.numSubDomainsY = numSubDomainsY;
    subDomain.numSubDomainsZ = numSubDomainsZ;
    subDomain.numSubDomains  = numSubDomains ;
    subDomain.numSubPointsX  = numSubPointsX ;
    subDomain.numSubPointsY  = numSubPointsY ;
    subDomain.numSubPointsZ  = numSubPointsZ ;
    subDomain.numSubPoints   = numSubPoints  ;
    //save values in furtures and in optimizationData
    subDomain.subMatrix_f = hpx::make_ready_future(std::move(subMatrix));

    //save geometry information and data to SubVectors
    SubVector subB;
    SubVector subX;
    SubVector subR;
    SubVector subZ;
    SubVector subP;
    SubVector subAp;
    // length
    subB.subLength = numSubPoints;
    subX.subLength = numSubPoints;
    subR.subLength = numSubPoints;
    subZ.subLength = numSubPoints;
    subP.subLength = numSubPoints;
    subAp.subLength = numSubPoints;
    // ID
    subB.subID = subID;
    subX.subID = subID;
    subR.subID = subID;
    subZ.subID = subID;
    subP.subID = subID;
    subAp.subID = subID;
    // Values
    subB.subValues_f = hpx::make_ready_future(std::move(subBv));
    subX.subValues_f = hpx::make_ready_future(std::move(subXv));
    subR.subValues_f = hpx::make_ready_future(std::move(subRv));
    subZ.subValues_f = hpx::make_ready_future(std::move(subZv));
    subP.subValues_f = hpx::make_ready_future(std::move(subPv));
    subAp.subValues_f = hpx::make_ready_future(std::move(subApv));    

    // save the SubDomain and SubVectors in the optimizationData structure
    subDomains.push_back( std::move(subDomain) );
    subBs.push_back( std::move(subB) );
    subXs.push_back( std::move(subX) );
    subRs.push_back( std::move(subR) );
    subZs.push_back( std::move(subZ) );
    subPs.push_back( std::move(subP) );
    subAps.push_back( std::move(subAp) );

/**************MULTIGRID*******************************************************/
/* TODO anpassen auf neue Daten Struktur
    // go throu MG Levels
    SparseMatrix* Acur = &A;
    local_int_t numSubPointsxc = numSubPointsx;
    local_int_t numSubPointsyc = numSubPointsy;
    local_int_t numSubPointszc = numSubPointsz;


    while (0 != Acur->Ac){
        // TODO dependencie matrix for MG
        numSubPointsxc /= 2;
        numSubPointsyc /= 2;
        numSubPointszc /= 2;
        local_int_t numberCourseRows = numSubPointsxc*numSubPointsyc*numSubPointszc;
        assert(numSubPointsxc > 0 && numSubPointsyc > 0 && numSubPointszc > 0);

        SubDomain subCDomain;
        SubF2COperator subF2C;

        // inizalize data pointers Martix
        SubMatrix cMatrix;
        std::vector<double*> & csubMatrix   = cMatrix.values;
        std::vector<double*> & cmatrixDiagonal = cMatrix.diagonal;
        std::vector<global_int_t*> & cMtxIndG  = cMatrix.indG;
        std::vector<local_int_t*> & cMtxIndLoc = cMatrix.indLoc;
        std::vector<char*> & cNonzerosInRow    = cMatrix.nonzerosInRow;

        csubMatrix.reserve(numberCourseRows);
        cmatrixDiagonal.reserve(numberCourseRows);
        cMtxIndG.reserve(numberCourseRows);
        cMtxIndLoc.reserve(numberCourseRows);
        cNonzerosInRow.reserve(numberCourseRows);

        // ... Vectors
        std::vector<double*> subRv;
        std::vector<double*> subXv;
        std::vector<double*> subAxfv;

        subRv.reserve(numberCourseRows);
        subXv.reserve(numberCourseRows);
        subAxfv.reserve(numberCourseRows);

        // ... F2C Opertaror
        std::vector<local_int_t*> & subF2CrOp   = subF2C.f2cOperator;
        subF2CrOp.reserve(numberCourseRows);

        // compute pointers from SubMatrixes/SubVectors to local(ety) matrix/vectors
        for (local_int_t iz=0; iz<numSubPointszc; iz++) {
            global_int_t giz = subIDz*numSubPointszc+iz;
            for (local_int_t iy=0; iy<numSubPointsyc; iy++) {
                global_int_t giy = subIDy*numSubPointsyc+iy;
                for (local_int_t ix=0; ix<numSubPointsxc; ix++) {
                    global_int_t gix = subIDx*numSubPointsxc+ix;

                    local_int_t currentLocalSubRow  = iz*numSubPointsxc*numSubPointsyc+iy*numSubPointsxc+ix;
                    global_int_t currentLocaletyRow = giz*nx*ny+giy*nx+gix;

                    csubMatrix.push_back(
                        Acur->subMatrix[currentLocaletyRow]);
                    cmatrixDiagonal.push_back(
                        Acur->matrixDiagonal[currentLocaletyRow]);
                    cMtxIndG.push_back(
                        Acur->mtxIndG[currentLocaletyRow]);
                    cMtxIndLoc.push_back(
                        Acur->mtxIndL[currentLocaletyRow]);
                    cNonzerosInRow.push_back(
                        &Acur->nonzerosInRow[currentLocaletyRow]);

                    subF2CrOp.push_back(
                        &Acur->mgData->f2cOperator[currentLocaletyRow]);

                    subRv.push_back(
                        &Acur->mgData->rc->values[currentLocaletyRow]);
                    subXv.push_back(
                        &Acur->mgData->xc->values[currentLocaletyRow]);
                    subAxfv.push_back(
                        &Acur->mgData->Axf->values[currentLocaletyRow]);
                }
            }
        }

        // save gernerl geometry information to SubDomain
        subCDomain.localNumberRows = numberCourseRows;
        subCDomain.nnumSubDomainsX = nnumSubDomainsX;
        subCDomain.numSubDomainsY = numSubDomainsY;
        subCDomain.numSubDomainsZ = numSubDomainsZ;
        subCDomain.numSubDomains  = numSubDomains ;
        subCDomain.numSubPointsX  = numSubPointsxc;
        subCDomain.numSubPointsY  = numSubPointsyc;
        subCDomain.numSubPointsZ  = numSubPointszc;
        subCDomain.subMatrix = hpx::make_ready_future(std::move(cMatrix));

        // save values to vector
        SubVector subR;
        SubVector subX;
        SubVector subAxf;

        subR.localLength   = numberCourseRows;
        subX.localLength   = numberCourseRows;
        subAxf.localLength = numberCourseRows;
        subR.localetyValues = Acur->mgData->rc->values;
        subX.localetyValues = Acur->mgData->xc->values;
        subAxf.localetyValues = Acur->mgData->Axf->values;
        subR.subValues_f = hpx::make_ready_future(std::move(subRv));
        subX.subValues_f = hpx::make_ready_future(std::move(subXv));
        subAxf.subValues_f = hpx::make_ready_future(std::move(subAxfv));

        // save in locality Vector
        static_cast< std::vector<SubDomain>* >(Acur->Ac->optimizationData)
            ->push_back(std::move(subCDomain));

        static_cast< std::vector<SubF2C>*>(Acur->mgData->optimizationData)
            ->push_back( hpx::make_ready_future(std::move(subF2C)) );

        static_cast< std::vector<SubVector>*>(Acur->mgData->rc->optimizationData)
            ->push_back(std::move(subR));
        static_cast< std::vector<SubVector>*>(Acur->mgData->xc->optimizationData)
            ->push_back(std::move(subX));
        static_cast< std::vector<SubVector>*>(Acur->mgData->Axf->optimizationData)
            ->push_back(std::move(subAxf));

        Acur = Acur->Ac;
    }
*/
}

return 0;
#else   // HPCG_NOHPX
return 0;
#endif  // HPCG_NOHPX
}
