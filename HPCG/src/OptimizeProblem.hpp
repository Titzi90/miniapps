
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

#include <algorithm>
#include <cassert>
#include <vector>
#include <set>

#ifndef HPCG_NOHPX
#include <hpx/hpx.hpp>


/*******************MATRIX*****************************************************/

// SubMatrix holds the matrix values of the SubDomain
struct SubMatrix{
    // Vector of pointers to original matrix rows
    std::vector<double*> values;
    // Vector of pointers to diagonal value
    std::vector<double*> diagonal;
    // Pointer to matrix indexi of the global domain
    //TODO zur Zeit eh nur eine Localety, danach muss man mal schauen wie das läuft.
    std::vector<global_int_t*> indG;
    // Vector of pointers to original matrix indexi in the localyty domain
    std::vector<local_int_t*> indLoc;
    // Vector of pointers to the number of nonzero Values per row
    std::vector<char*> nonzerosInRow;
};
typedef hpx::shared_future<SubMatrix> SubMatrix_future;

// SubDomain holdes the SubMatrix and geometic datas
struct SubDomain{
    //TODO auf mehren Localitys? Reicht es dafür zu soregen das auf localer Locelety ausgeführt wird?
    int numSubDomains ;     // Number of SubDomains
    int numSubDomainsX;     // Number of SubDomains in x direction.
    int numSubDomainsY;     // Number of SubDomains in y direction.
    int numSubDomainsZ;     // Number of SubDomains in z direction.
    int subID ;     // SubDomain ID
    int subIDx;     // SubDomain ID in x direction
    int subIDy;     // SubDomain ID in y direction
    int subIDz;     // SubDomain ID in z direction
    int numSubPoints ;       // number of points in SubDomain
    int numSubPointsX;       // number of points in x direction
    int numSubPointsY;       // number of points in y direction
    int numSubPointsZ;       // number of points in z direction

    std::set<int> dependencies; // set of dependent SubDomainIDs (subID)

    SubMatrix_future subMatrix_f;   // Future of SubMatrix holding the values
};


/*******************VECTOR*****************************************************/

// SubVectorValues holdes the vector values of the SubDomain
typedef std::vector<double*> SubVectorValues;
typedef hpx::shared_future<SubVectorValues> SubVectorValues_future;

// SubVector holdes the SubVectorValues and geometry date and meta informations
struct SubVector{
    int subID;          // corresponding SubDomain ID
    int subLength;      // Number of values in subDomain
    SubVectorValues_future subValues_f;     // future of the subVectorValues

};

/*******************MG OPERATOR************************************************/

typedef std::vector<local_int_t*> SubF2COperatorValues;
typedef hpx::shared_future<SubF2COperatorValues> SubF2COperatorValues_f;


/*******************HELPER FUNKTIONS*******************************************/

// calculating the 1D SubDomain index out of the 3D index
inline int index(int const subIDx, int const subIDy, int const subIDz,
                 int const numSubDomainsX, int const numSubDomainsY,
                 int const numSubDomainsZ)
{
    return subIDz * (numSubDomainsX*numSubDomainsY) + subIDy * (numSubDomainsX) + subIDx;
}

// calculating the 1D prozessor index from a SubDomain
inline int index(SubDomain const & A){
    return index(A.subIDx, A.subIDy, A.subIDz,
                 A.numSubDomainsX, A.numSubDomainsY, A.numSubDomainsZ);
}

// calculating the SubDomain (1D index) that conatins the element
// giffen by the (localety) matrix index
// TODO mehre localetys???
inline int belongsTo (const int element,
                      const int numSubDomainsX,
                      const int numSubDomainsY,
                      const int numSubDomainsZ,
                      const int numSubPointsX,
                      const int numSubPointsY,
                      const int numSubPointsZ
                     )
{ 
    int nz = numSubDomainsZ * numSubPointsZ;
    int ny = numSubDomainsY * numSubPointsY;
    int nx = numSubDomainsX * numSubPointsX;

    // calculate the 3D cordinates of the element
    int eleZ =  element / (nx*ny);
    int eleY = (element-eleZ*nx*ny) / nx;
    int eleX =  element % nx;

    // calculate the 3d cordinates of the containing SubDomain
    int subZ = eleZ / numSubPointsZ;
    int subY = eleY / numSubPointsY;
    int subX = eleX / numSubPointsX;

    // calculate the 1D index of the subDomain
    return index(subX, subY, subZ,
                 numSubDomainsX, numSubDomainsY, numSubDomainsZ);
}

// returens a future that becomes ready when the Vector v is ready
// this is when all values in the SubVectors are ready
inline hpx::future<std::vector<SubVectorValues_future> > when_vec(Vector v)
{
    std::vector<SubVectorValues_future> subVVs_f;
    
    std::vector<SubVector>  & subVs =
        *static_cast<std::vector<SubVector>* >(v.optimizationData);
    for(size_t i=0; i<subVs.size(); ++i)
    {
        subVVs_f.push_back(subVs.at(i).subValues_f);
    }

    return hpx::when_all(subVVs_f);
}

#endif  // NOT NO_HPX

// Funtion to create Optimized Data
int OptimizeProblem(SparseMatrix & A,
                    CGData & data,
                    Vector & b,
                    Vector & x,
                    Vector & xexact
                   );

#endif  // OPTIMIZEPROBLEM_HPP
