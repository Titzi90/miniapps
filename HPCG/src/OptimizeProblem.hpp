
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

#ifndef OPTIMIZEPROBLEM_HPP
#define OPTIMIZEPROBLEM_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"

#ifndef HPCG_NOHPX

#include <hpx/hpx.hpp>



//TODO daten thread save machen
struct LocalSubDomain_STRUCT {
    //TODO constant geo values out of future
    int nlpx;       // number of local sub process in x direction.
    int nlpy;       // number of local sub process in y direction.
    int nlpz;       // number of local sub process in z direction.
    int nlp ;       // number of locasl sub process in this locality
    int lpix;       // local process id in x direction
    int lpiy;       // local process id in x direction
    int lpiz;       // local process id in x direction
    int nlx ;       // number of local points in x direction
    int nly ;       // number of local points in y direction
    int nlz ;       // number of local points in z direction
    local_int_t localNumberRows;

    double ** matrixValues;
    double ** matrixDiagonal;
    global_int_t ** mtxIndG;   // Pointer to matrix indexi of the global domain
    local_int_t ** mtxIndLoc;  // Pointer to matrix indexi of the localyty domain
    char * nonzerosInRow;

};
typedef struct LocalSubDomain_STRUCT LocalSubDomain;
typedef hpx::shared_future<LocalSubDomain> SubDomain_future;

/******************************************************************************/

struct LocalSubVector_STRUCT {
    local_int_t localLength;
    double * values; 

};
typedef struct LocalSubVector_STRUCT LocalSubVector;
typedef hpx::shared_future<LocalSubVector> SubVector_future;

/******************************************************************************/

struct LocalSubF2C_STRUCT {
    local_int_t * f2cOperator;
};
typedef struct LocalSubF2C_STRUCT LocalSubF2C;
typedef hpx::shared_future<LocalSubF2C> SubF2C_future;

/******************************************************************************/

inline int index(int const lpix, int const lpiy, int const lpiz,
                 int const nlpx, int const nlpy, int const nlpz){
    return lpiz * (nlpx*nlpy) + lpiy * (nlpx) + lpix;
}

inline int index(LocalSubDomain const & A){
    return index(A.lpix, A.lpiy, A.lpiz, A.nlpx, A.nlpy, A.nlpz);
}


struct Neighborhood{
    Neighborhood(){
        for (int i=0;i<27;++i)
            *(data+i) = -1;
    }

    void insert(int const x, int const y, int const z, int const val){
        asserte(x >=-1 && x<=1 && y >=-1 && y<=1 && z >=-1 && z<=1);
        data[x+1][y+1][z+1] = val;
    }

    int get(int const x, int const y, int const z) const{
        asserte(x >=-1 && x<=1 && y >=-1 && y<=1 && z >=-1 && z<=1);
        return data[x+1][y+1][z+1];
    }

    bool hasNeighbor(int const x, int const y, int const z) const{
        asserte(x >=-1 && x<=1 && y >=-1 && y<=1 && z >=-1 && z<=1);
        return data[x+1][y+1][z+1] != -1;
    }

    std::vector<int> getNeighbors(){
        std::vector<int> neighbors;
        neighbors.reserve(27);
        
        for (int i=0; i<27; ++i){
            if(*(data+i) != -1)
                neighbors.push_back( *(data+i) );
        }

        return neighbors;
    }

private:
    int [3][3][3] data;
};

struct Neighborhood_futuriced{
    Neighborhood_futuriced(){
    }

    hpx::shared_future<double*>*** & get(){
        return data;
    }

    shared_future<double*>& get(int const x, int const y, int const z){
        asserte(x >=-1 && x<=1 && y >=-1 && y<=1 && z >=-1 && z<=1);
        return data[x+1][y+1][z+1];
    }

    void set(int const x, int const y, int const z,
             hpx::shared_future<double*> values){
        asserte(x >=-1 && x<=1 && y >=-1 && y<=1 && z >=-1 && z<=1);

        data[x+1][y+1][z+1] = values;
    }

private:
    hpx::shared_future<double*>[3][3][3] data;
};


//TODO Hallo
Neighborhood getNeighbor(LocalSubDomain const & A){
    int& lpix = A.lpix;
    int& lpiy = A.lpiy;
    int& lpiz = A.lpiz;
    int& nlpx = A.nlpx;
    int& nlpy = A.nlpy;
    int& nlpz = A.nlpz;

    Neighborhood neighborhod();

    for (int dx=-1;dx<=1;++dx){
        int x = dx + lpix;
        if(x<0 || x>=nlpx) continue;

        for (int dy=-1;dy<=1;++dy){
            int y = dy + lpiy;
            if(y<0 || y>=nlpy) continue;

            for (int dz=-1;dz<=1;++dz){
                int z = dz + lpiz;
                if(y<0 || y>=nlpy) continue;

                neighborhod.insert(dx,dy,dz, index(x,y,z,nlpx,nlpy,nlpz) ); 
            }
        }
    }

    return neighborhod;
}



int OptimizeProblem(SparseMatrix & A, CGData & data,  Vector & b, Vector & x, Vector & xexact);

#endif  // NOT NO_HPX
#endif  // OPTIMIZEPROBLEM_HPP
