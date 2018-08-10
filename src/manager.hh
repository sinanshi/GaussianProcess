#ifndef __managerH
#define __managerH

#include <assert.h>
#include <iostream>
//#include <cusolverDn.h>
//#include <cuda_runtime.h>
#include <magma_v2.h>
#include <iostream>
using namespace std;

#define GPREAL double

class gpuPrepareLikelihood {
  private:
    magma_queue_t queue;
    magma_int_t err;
    magma_int_t info;

    GPREAL *dev_Q; 
    GPREAL *dev_targets;
    GPREAL *dev_invQt;



  public:
    GPREAL *Q; //! input matrix
    GPREAL *targets; //! input vector
    GPREAL *invQ; //! the inverse matrix  (L^T)^-1(L)^-1
    GPREAL *invQt;  //! the solution vector
    //GPREAL *logdetQ; //! for posterior log likelihood.

//    GPREAL *L; 
    int N; //! dimention

    GPREAL logdetQ;

    //! constructor - initialise CUDA handlers
    gpuPrepareLikelihood(GPREAL *Q_, GPREAL *targets_, int N); 
    ~gpuPrepareLikelihood(); 

    void gpu_cholesky();


};


#endif
