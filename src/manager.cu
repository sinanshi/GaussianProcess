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
  targets = targets_;
  N = N_;
  logdetQ = 0; 

  //! initialise magma
  magma_init(); 
  magma_int_t dev = 0;
  magma_queue_create(dev, &queue);

  magma_smalloc_cpu(&invQ, N * N); 
  magma_smalloc_cpu(&invQt, N);
  
  magma_smalloc(&dev_Q, N * N);
  magma_smalloc(&dev_targets, N);
  magma_smalloc(&dev_invQt, N);

}



gpuPrepareLikelihood::~gpuPrepareLikelihood(){
  //! free the memory
  magma_free(dev_Q);
  magma_free(dev_targets);

  free(invQ);
  free(invQt);
  free(targets);
  //! finalize magam
  magma_queue_destroy(queue);
  magma_finalize(); 
}



void gpuPrepareLikelihood::gpu_cholesky() {
  magma_ssetmatrix(N, N, Q, N, dev_Q, N, queue);
  magma_spotrf_gpu(MagmaLower, N, dev_Q, N, &info);

  // have to write a kernel latter to avoid copyting
  //-------------------------------------------------
  magma_sgetmatrix(N, N, dev_Q, N, invQ, N, queue);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (j < i) invQ[ind(i, j, N)] = 0;
      if (i == j) logdetQ += 2.0 * log(invQ[ind(i, j, N)]);
    }
  }
  magma_ssetmatrix(N, N, invQ, N, dev_Q, N, queue);
  //-------------------------------------------------
  
  magma_strtri_gpu(MagmaLower, MagmaNonUnit, N, dev_Q, N, &info);
  magma_slauum_gpu(MagmaLower, N, dev_Q, N, &info);// L^T L
  
  magma_sgetmatrix(N, N, dev_Q, N, invQ, N, queue); //invQ

  // here maybe to kernels too? or maybe not?
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      invQ[ind(j, i, N)] = invQ[ind(i, j, N)];
    }
  }

  // calculate the solution of the tridiagnal system invQt
  magma_ssetvector(N, targets, 1, dev_targets, 1, queue);
  magma_ssymv(MagmaLower, N, 1, dev_Q, N, dev_targets, 1, 
      0, dev_invQt, 1, queue);
  magma_sgetvector(N, dev_invQt, 1, invQt, 1, queue);

}


