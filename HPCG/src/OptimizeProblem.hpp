
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
#include <cassert>
#include <vector>

#ifndef HPCG_NOHPX

#include <hpx/hpx.hpp>


/*******************MATRIX*****************************************************/

// datastructure of Matrixvalues
struct MatrixValues{
    std::vector<double*> values;
    std::vector<double*> diagonal;
    std::vector<global_int_t*> indG;   // Pointer to matrix indexi of the global domain TODO ???
    std::vector<local_int_t*> indLoc;  // Pointer to matrix indexi of the localyty domain
    std::vector<char*> nonzerosInRow;
};
typedef hpx::shared_future<MatrixValues> MatrixValues_furure;


// SubMatrix including values and geometic data
struct SubDomain{
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

    MatrixValues_furure matrixValues;
};


/*******************VECTOR*****************************************************/

typedef hpx::shared_future< std::vector<double*> > VectorValues_future;

// dummy ready future for non existing neighbors
static hpx::shared_future< std::vector<double*> > dummyNeighbor_f =
    hpx::make_ready_future(std::vector<double*>() );

// subvector including data and geometry and meta informations
struct SubVector{
    // the lenth of the subvector
    local_int_t localLength;
    
    // pointer to the whol vector on the localety
    double* localetyValues;

    // future of the subvectorvalues (subvector points to localety values)
    VectorValues_future values_f;

    // pointers too all sourunding fututres of subvectorvalues.
    // TODO in vector
    VectorValues_future* neighbourhood[3][3][3];

    std::vector<hpx::shared_future< std::vector<double*> > > getNeighbourhood() const {
        std::vector<hpx::shared_future< std::vector<double*> > > neighbours;
        neighbours.reserve(9);
        for (int x=0; x<3; ++x){
            for (int y=0; y<3; ++y){
                for (int z=0; z<3; ++z){
                    neighbours.push_back(*neighbourhood[x][y][z]);
                }
            }
        }

        return std::move(neighbours);
    }

};


/*******************MG OPERATOR************************************************/

struct SubF2COperator{
    std::vector<local_int_t*> f2cOperator;
};
typedef hpx::shared_future<SubF2COperator> SubF2C;


/*******************HELPER FUNKTIONS*******************************************/

// ca)culating the 1D prozessor index out of the 3D index
inline int index(int const lpix, int const lpiy, int const lpiz,
                 int const nlpx, int const nlpy, int const nlpz){
    return lpiz * (nlpx*nlpy) + lpiy * (nlpx) + lpix;
}

// calculating the 1D prozessor index from a SubDomain
inline int index(SubDomain const & A){
    return index(A.lpix, A.lpiy, A.lpiz, A.nlpx, A.nlpy, A.nlpz);
}

//wait untill SubDomain is ready
inline hpx::future<std::vector<VectorValues_future> > when_vec(Vector v)
{
    std::vector<VectorValues_future> subVs_f;
    
    std::vector<SubVector>  & subVs =
        *static_cast<std::vector<SubVector>* >(v.optimizationData);
    for(size_t i=0; i<subVs.size(); ++i)
    {
        subVs_f.push_back(subVs.at(i).values_f);
    }

    return hpx::when_all(subVs_f);
}


//TODO brauch ich das noch?
/*
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
    int data[3][3][3];
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
*/

// Funtion to create Optimized Data
int OptimizeProblem(SparseMatrix & A, CGData & data,  Vector & b, Vector & x, Vector & xexact);

#else
// dummy function
int OptimizeProblem(SparseMatrix & A, CGData & data,  Vector & b, Vector & x, Vector & xexact);
#endif  // NOT NO_HPX
#endif  // OPTIMIZEPROBLEM_HPP
