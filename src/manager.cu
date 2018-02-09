#include <kernel.cu>
#include <manager.hh>

gpuPrepareLikelihood::gpuPrepareLikelihood(float *Q_, float *targets_, int N_) {
  //! initialise the input by pointing to the numpy 
  //! array passed into the wrapper. 
  Q = Q_;
  targets = targets_;
  N = N_; 

  //! initialise magma
  magma_init(); 
  magma_int_t dev = 0;
  magma_queue_create(dev, &queue);

//  magma_smalloc_cpu(&invQ, N * N);
//  magma_smalloc_cpu(&invQt, N);
//  magma_smalloc_cpu(logdetQ)
  
//  magma_smalloc_cpu(&targets, N * N); 
  magma_smalloc_cpu(&L, N * N); 
  magma_smalloc(&dev_Q, N * N);

  magma_smalloc_cpu(&invQt, N);


  
//  magma_smalloc(&dev_targets, N * N);
//  magma_ssetmatrix(N, N, Q, N, dev_Q, N, queue);
//  magma_ssetmatrix(N, 1, targets, N, dev_Q, N, queue);




}



gpuPrepareLikelihood::~gpuPrepareLikelihood(){

  //! finalize magam
  magma_free(dev_Q);

//  magma_free(dev_Q); 
//  magma_free(dev_targets);

  free(L);
  magma_queue_destroy(queue);
  magma_finalize(); 



}

__inline__ int ind(int r, int c, int n) {
  return(r * n + c); 
}

void gpuPrepareLikelihood::gpu_cholesky() {
  magma_ssetmatrix(N, N, Q, N, dev_Q, N, queue);
  magma_spotrf_gpu(MagmaLower, N, dev_Q, N, &info);

  // have to write a kernel latter to avoid copyting
  magma_sgetmatrix(N, N, dev_Q, N, L, N, queue);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (j < i) L[ind(i, j, N)] = 0;
    }
  }
  magma_ssetmatrix(N, N, L, N, dev_Q, N, queue);

  
  magma_strtri_gpu(MagmaLower, MagmaNonUnit, N, dev_Q, N, &info);
  magma_slauum_gpu(MagmaLower, N, dev_Q, N, &info);// L^T L
  
  magma_sgetmatrix(N, N, dev_Q, N, L, N, queue); //invQ

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      L[ind(j, i, N)] = L[ind(i, j, N)];
    }
  }


//  magma_sprint( 5, 5, Q, N);
//  magma_sprint( 5, 5, L, N);

}


