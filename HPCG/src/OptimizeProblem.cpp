
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
int OptimizeProblem(SparseMatrix & A,
                    CGData & data,
                    Vector & b,
                    Vector & x,
                    Vector & xexact)
{

// This function can be used to completely transform any part of the data structures.
// Right now it does nothing, so compiling with a check for unused variables results in complaints

#ifndef HPCG_NOHPX
    //TODO mit mehrern Localitys
    //TODO DEBUG output

#ifdef HPCG_DEBUG
    std::cerr << "start opti problem" << std::endl;
#endif

    //TODO CG DATA!!!

// extract datas
int const & nx = A.geom->nx;    // number points in x direction for the locality
int const & ny = A.geom->ny;    // number points in y direction for the locality
int const & nz = A.geom->nz;    // number points in z direction for the locality

// calc local subdomain geometry
// we need at least 16 points in each direction for MG
// TODO make it more dynamic
int const NL = 16;  //number of local points in each direction
assert(nx%NL == 0 && ny%NL == 0 && nz%NL == 0);
int nlpx = nx / NL;         // number of local sub prozessors in x direction.
int nlpy = ny / NL;         // number of local sub prozessors in y direction.
int nlpz = nz / NL;         // number of local sub prozessors in z direction.
int nlp  = nlpx * nlpy * nlpz;  // number of locasl sub prozessors
int nlx  = NL;              // number of local points in x direction
int nly  = NL;              // number of local points in y direction
int nlz  = NL;              // number of local points in z direction
local_int_t localNumberRows = nlx*nly*nlz;

// create an allocate Sub Domains
b.optimizationData = new std::vector<SubVector>();
x.optimizationData = new std::vector<SubVector>();
A.optimizationData = new std::vector<SubDomain>();

std::vector<SubVector> & subBs=
    *static_cast<std::vector<SubVector>* >(b.optimizationData);
std::vector<SubVector> & subXs=
    *static_cast<std::vector<SubVector>* >(x.optimizationData);
std::vector<SubDomain> & subDomains=
    *static_cast<std::vector<SubDomain>* >(A.optimizationData);

subBs.reserve(nlp);
subXs.reserve(nlp);
subDomains.reserve(nlp);

//create subDomains for MultiGrid
SparseMatrix& Acur = A;
while (0 != Acur.Ac){
    Acur.Ac->optimizationData          = new std::vector<SubDomain>();
    Acur.mgData->optimizationData      = new std::vector<SubF2C>();
    Acur.mgData->rc->optimizationData  = new std::vector<SubVector>();
    Acur.mgData->xc->optimizationData  = new std::vector<SubVector>();
    Acur.mgData->Axf->optimizationData = new std::vector<SubVector>();

    static_cast<std::vector<SubDomain>* >(Acur.Ac->optimizationData)
        ->reserve(nlp);
    static_cast<std::vector<SubF2C>* >(Acur.mgData->optimizationData)
        ->reserve(nlp);
    static_cast<std::vector<SubVector>* >(Acur.mgData->rc->optimizationData)
        ->reserve(nlp);
    static_cast<std::vector<SubVector>* >(Acur.mgData->xc->optimizationData)
        ->reserve(nlp);
    static_cast<std::vector<SubVector>* >(Acur.mgData->Axf->optimizationData)
        ->reserve(nlp);

    Acur = *(Acur.Ac);
}


// initialize SubDomains and SubVectors
//TODO parallel ACHTUNG ich verlasse mich auf reinfolge der subdomains
//->veilleicht eine map verwenden
for (int i=0; i<nlp; ++i){
    SubDomain subDomain;

    //compute x y z rank of subDomain/Vector (index i --> cordinate x,y,z)
    int & lpix = subDomain.lpix;
    int & lpiy = subDomain.lpiy;
    int & lpiz = subDomain.lpiz;

    lpiz = i/(nlpx*nlpy);
    lpiy = (i-lpiz*nlpx*nlpy)/nlpx;
    lpix = i%nlpx;

    // inizalize data pointers for Matrix
    MatrixValues matrix;
    double** & matrixValues = matrix.values;
    double** & matrixDiagonal = matrix.diagonal;
    global_int_t** & mtxIndG = matrix.indG;
    local_int_t** & mtxIndLoc = matrix.indLoc;
    char* & nonzerosInRow = matrix.nonzerosInRow;

    matrixValues   = new double*[localNumberRows];
    matrixDiagonal = new double*[localNumberRows];
    mtxIndG        = new global_int_t*[localNumberRows];
    mtxIndLoc      = new local_int_t*[localNumberRows];
    nonzerosInRow  = new char[localNumberRows];

    // .. for vector
    double* subBv[localNumberRows];
    double* subXv[localNumberRows];


    // compute pointers from SubMatrixes/SubVectors to local(ety) matrix/vectors
    for (local_int_t iz=0; iz<nlz; iz++) {
        global_int_t giz = lpiz*nlz+iz;
        for (local_int_t iy=0; iy<nly; iy++) {
            global_int_t giy = lpiy*nly+iy;
            for (local_int_t ix=0; ix<nlx; ix++) {
                global_int_t gix = lpix*nlx+ix;

                local_int_t currentLocalSubRow = iz*nlx*nly+iy*nlx+ix;
                global_int_t currentLocaletyRow = giz*nx*ny+giy*nx+gix;

                matrixValues[currentLocalSubRow] =
                    A.matrixValues[currentLocaletyRow];
                matrixDiagonal[currentLocalSubRow] =
                    A.matrixDiagonal[currentLocaletyRow];
                mtxIndG[currentLocalSubRow] =
                    A.mtxIndG[currentLocaletyRow];
                mtxIndLoc[currentLocalSubRow] =
                    A.mtxIndL[currentLocaletyRow];
                nonzerosInRow[currentLocalSubRow]=
                    A.nonzerosInRow[currentLocaletyRow];

                subBv[currentLocalSubRow] = &b.values[currentLocaletyRow];
                subXv[currentLocalSubRow] = &x.values[currentLocaletyRow];

            }
        }
    }

    // save gernerl geometry information to SubDomain on uper level (A)
    subDomain.localNumberRows = localNumberRows;
    subDomain.nlpx = nlpx;
    subDomain.nlpy = nlpy;
    subDomain.nlpz = nlpz;
    subDomain.nlp  = nlp ;
    subDomain.nlx  = nlx ;
    subDomain.nly  = nly ;
    subDomain.nlz  = nlz ;
    // TODO lpix/y/z speichern
    //save values in furtures and in optimizationData
    subDomain.matrixValues = hpx::make_ready_future(std::move(matrix));

    // save values to vector
    SubVector subB;
    SubVector subX;

    subB.localLength = localNumberRows;
    subX.localLength = localNumberRows;
    subB.localetyValues = b.values;
    subX.localetyValues = x.values;
    subB.values_f = hpx::make_ready_future<double**>(subBv);
    subX.values_f = hpx::make_ready_future<double**>(subXv);

    //save in SubParts in Loacality Vector
    // TODO wenn map dann inert
    subBs.push_back( std::move(subB) );
    subXs.push_back( std::move(subX) );
    subDomains.push_back( subDomain );


    // go throu MG Levels
    SparseMatrix& Acur = A;
    local_int_t nlxc = nlx;
    local_int_t nlyc = nly;
    local_int_t nlzc = nlz;

    while (0 != Acur.Ac){
        nlxc /= 2;
        nlyc /= 2;
        nlzc /= 2;
        local_int_t numberCourseRows = nlxc*nlyc*nlzc;
        assert(nlxc > 0 && nlyc > 0 && nlzc > 0);

        SubDomain subCDomain;
        SubF2COperator subF2C;

        // inizalize data pointers Martix
        MatrixValues cMatrix;
        double ** & cmatrixValues   = cMatrix.values;
        double ** & cmatrixDiagonal = cMatrix.diagonal;
        global_int_t ** & cMtxIndG  = cMatrix.indG;
        local_int_t ** & cMtxIndLoc = cMatrix.indLoc;
        char * & cNonzerosInRow     = cMatrix.nonzerosInRow;

        cmatrixValues   = new double*[numberCourseRows];
        cmatrixDiagonal = new double*[numberCourseRows];
        cMtxIndG        = new global_int_t*[numberCourseRows];
        cMtxIndLoc      = new local_int_t*[numberCourseRows];
        cNonzerosInRow  = new char[numberCourseRows];

        // ... Vectors
        double* subRv[numberCourseRows];
        double* subXv[numberCourseRows];
        double* subAxfv[numberCourseRows];

        // ... F2C Opertaror
        local_int_t** & subF2CrOp   = subF2C.f2cOperator;
        subF2CrOp       = new local_int_t*[numberCourseRows];

        // compute pointers from SubMatrixes/SubVectors to local(ety) matrix/vectors
        for (local_int_t iz=0; iz<nlzc; iz++) {
            global_int_t giz = lpiz*nlzc+iz;
            for (local_int_t iy=0; iy<nlyc; iy++) {
                global_int_t giy = lpiy*nlyc+iy;
                for (local_int_t ix=0; ix<nlxc; ix++) {
                    global_int_t gix = lpix*nlxc+ix;

                    local_int_t currentLocalSubRow  = iz*nlxc*nlyc+iy*nlxc+ix;
                    global_int_t currentLocaletyRow = giz*nx*ny+giy*nx+gix;

                    cmatrixValues[currentLocalSubRow] =
                        Acur.matrixValues[currentLocaletyRow];
                    cmatrixDiagonal[currentLocalSubRow] =
                        Acur.matrixDiagonal[currentLocaletyRow];
                    cMtxIndG[currentLocalSubRow] =
                        Acur.mtxIndG[currentLocaletyRow];
                    cMtxIndLoc[currentLocalSubRow] =
                        Acur.mtxIndL[currentLocaletyRow];
                    cNonzerosInRow[currentLocalSubRow]=
                        Acur.nonzerosInRow[currentLocaletyRow];

                    subF2CrOp[currentLocalSubRow]=
                        &Acur.mgData->f2cOperator[currentLocaletyRow];

                    subRv[currentLocalSubRow] =
                        &Acur.mgData->rc->values[currentLocaletyRow];
                    subXv[currentLocalSubRow] =
                        &Acur.mgData->xc->values[currentLocaletyRow];
                    subAxfv[currentLocalSubRow] =
                        &Acur.mgData->Axf->values[currentLocaletyRow];
                }
            }
        }

        // save gernerl geometry information to SubDomain
        subCDomain.localNumberRows = numberCourseRows;
        subCDomain.nlpx = nlpx;
        subCDomain.nlpy = nlpy;
        subCDomain.nlpz = nlpz;
        subCDomain.nlp  = nlp ;
        subCDomain.nlx  = nlxc;
        subCDomain.nly  = nlyc;
        subCDomain.nlz  = nlzc;
        // TODO lpix/y/z speichern
        subCDomain.matrixValues = hpx::make_ready_future(std::move(cMatrix));

        // save values to vector
        SubVector subR;
        SubVector subX;
        SubVector subAxf;

        subR.localLength   = numberCourseRows;
        subX.localLength   = numberCourseRows;
        subAxf.localLength = numberCourseRows;
        subR.localetyValues = Acur.mgData->rc->values;
        subX.localetyValues = Acur.mgData->xc->values;
        subAxf.localetyValues = Acur.mgData->Axf->values;
        subR.values_f = hpx::make_ready_future<double**>(std::move<double**>subRv);
        subX.values_f = hpx::make_ready_future<double**>(subXv);
        subAxf.values_f = hpx::make_ready_future<double**>(subAxfv);

        // save in locality Vector
        //TODO map ersetzen
        static_cast< std::vector<SubDomain>* >(Acur.Ac->optimizationData)
            ->push_back(std::move(subCDomain));

        static_cast< std::vector<SubF2C>*>(Acur.mgData->optimizationData)
            ->push_back( hpx::make_ready_future(std::move(subF2C)) );

        static_cast< std::vector<SubVector>*>(Acur.mgData->rc->optimizationData)
            ->push_back(std::move(subR));
        static_cast< std::vector<SubVector>*>(Acur.mgData->xc->optimizationData)
            ->push_back(std::move(subX));
        static_cast< std::vector<SubVector>*>(Acur.mgData->Axf->optimizationData)
            ->push_back(std::move(subAxf));

        Acur = *(Acur.Ac);
    }
}

//TODO MGdata
// set naighbour futures to Vectors
//TODO more localetys
//loop ofer all elements
for (int x=0; x<nlpx; ++x){
    for (int y=0; y<nlpy; ++y){
        for (int z=0; z<nlpz; ++z){

            int i = index(x,y,z,nlpx,nlpy,nlpz);

            //loop over neighbors
            for (int dx=-1; dx<=1; ++dx){
                int xx = x + dx;
                for (int dy=-1; dy<=1; ++dy){
                    int yy = y + dy;
                    for (int dz=-1; dz<=1; ++dz){
                        int zz = z + dz;

                        //check if neighbor is a legal
                        if (xx<0 || xx>=nlpx ||
                            yy<0 || yy>=nlpy ||
                            zz<0 || zz>=nlpz)
                        {
                            //no neighbor -> set reddy dummy
                            subBs[i].neighbourhood[dx+1][dy+1][dz+1] =
                                &dummyNeighbor;
                            subXs[i].neighbourhood[dx+1][dy+1][dz+1] =
                                &dummyNeighbor;

//TODO MG


                        }
                        else{
                            int ii = index(xx,yy,zz,nlpx,nlpy,nlpz);
                            // link to neighbor values
                            subBs[i].neighbourhood[dx+1][dy+1][dz+1] =
                                &subBs[ii].values_f;
                            subXs[i].neighbourhood[dx+1][dy+1][dz+1] =
                                &subXs[ii].values_f;



//TODO MG



                        }
                    }
                }
            }
        }
    }
}





return 0;
#else
  return 0;
#endif
}
