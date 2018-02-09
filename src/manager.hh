#ifndef __managerH
#define __managerH

#include <assert.h>
#include <iostream>
//#include <cusolverDn.h>
//#include <cuda_runtime.h>
#include <magma_v2.h>
#include <iostream>
using namespace std;

class gpuPrepareLikelihood {
  private:
    magma_queue_t queue;
    magma_int_t err;
    magma_int_t info;

    float *dev_Q; 
    float *dev_targets;



  public:
    float *Q; //! input matrix
    float *targets; //! input vector
    float *invQ; //! the inverse matrix  (L^T)^-1(L)^-1
    float *invQt;  //! the solution vector
    //float *logdetQ; //! for posterior log likelihood.

    float *L; 
    int N; //! dimention

    //! constructor - initialise CUDA handlers
    gpuPrepareLikelihood(float *Q_, float *targets_, int N); 
    ~gpuPrepareLikelihood(); 

    void gpu_cholesky();


};


#endif
