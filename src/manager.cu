#include <kernel.cu>
#include <manager.hh>

//! Convert the 1D index to 2D.
__inline__ int ind(int r, int c, int n) {
  return(r * n + c); 
}


gpuPrepareLikelihood::gpuPrepareLikelihood(GPREAL *Q_, GPREAL *targets_, int N_) {
  //! initialise the input by pointing to the numpy 
  //! array passed into the wrapper. 

  Q = Q_;
//  targets = targets_;
  N = N_;
  logdetQ = 0;

  //! initialise magma
  magma_init(); 
  magma_int_t dev = 0;
  magma_queue_create(dev, &queue);

  magma_dmalloc_cpu(&invQ, N * N); 
  magma_dmalloc_cpu(&invQt, N);
  
  magma_dmalloc(&dev_Q, N * N);
  magma_dmalloc(&dev_targets, N);
  magma_dmalloc(&dev_invQt, N);

  targets = (GPREAL *)malloc(N * sizeof(GPREAL)); 
  for (int i = 0; i < N; ++i)
    targets[i] = targets_[i];

/*  Q = (GPREAL *)malloc(N * N * sizeof(GPREAL));
  for (int i = 0; i < N; ++i)
    Q[i] = Q_[i];
*/
}



gpuPrepareLikelihood::~gpuPrepareLikelihood(){
  //! free the memory
  magma_free(dev_Q);
  magma_free(dev_targets);

  free(invQ);
  free(invQt);
//  free(targets);
//  free(Q);
  //! finalize magam
  magma_queue_destroy(queue);
  magma_finalize(); 
}



void gpuPrepareLikelihood::gpu_cholesky() {
  magma_dsetmatrix(N, N, Q, N, dev_Q, N, queue);
  magma_dpotrf_gpu(MagmaLower, N, dev_Q, N, &info);

  // have to write a kernel latter to avoid copyting
  //-------------------------------------------------
  magma_dgetmatrix(N, N, dev_Q, N, invQ, N, queue);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (j < i) invQ[ind(i, j, N)] = 0;
      if (i == j) logdetQ += 2.0 * log(invQ[ind(i, j, N)]);
    }
  }
  magma_dsetmatrix(N, N, invQ, N, dev_Q, N, queue);
  //-------------------------------------------------
  magma_dtrtri_gpu(MagmaLower, MagmaNonUnit, N, dev_Q, N, &info);
  magma_dlauum_gpu(MagmaLower, N, dev_Q, N, &info);// L^T L
  magma_dgetmatrix(N, N, dev_Q, N, invQ, N, queue); //invQ

  // since the result is lower triangular matrix, the upper 
  // trangle is absent. Here is to patch the upper part of the 
  // matrix. 
  // here maybe to kernels too? or maybe not?
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      invQ[ind(j, i, N)] = invQ[ind(i, j, N)];
    }
  }

  // calculate the solution of the tridiagnal system invQt
  magma_dsetvector(N, targets, 1, dev_targets, 1, queue);
  magma_dsymv(MagmaLower, N, 1, dev_Q, N, dev_targets, 1,
      0, dev_invQt, 1, queue);
  magma_dgetvector(N, dev_invQt, 1, invQt, 1, queue);
}


