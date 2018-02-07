/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
#include <manager.hh>

gpuPrepareLikelihood::gpuPrepareLikelihood(float *Q_, float *targets_, int N) {
      cudaStatus = cudaGetDevice(0);
      cusolverStatus = cusolverDnCreate(&handle);
      cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

      Q = (float *)malloc(sizeof(float) * N * N);
      targets = (float *)malloc(sizeof(float) * N);

      Q = Q_; // point to Q
      targets = targets_;

      cudaStatus = cudaMalloc((void **) &d_Q, N * N * sizeof(float));
      cudaStatus = cudaMalloc((void **) &d_targets, N * sizeof(float));
//      cudaStatus = cudaMalloc((void **) &d_invQ, N * sizeof(float));
//      cudaStatus = cudaMalloc((void **) &d_info, sizeof(int));
      cudaStatus = cudaMemcpy(d_Q, Q, N * N * sizeof(float), cudaMemcpyHostToDevice); 
      cudaStatus = cudaMemcpy(d_targets, targets, N * sizeof(float), cudaMemcpyHostToDevice); 
}


gpuPrepareLikelihood::~gpuPrepareLikelihood(){

//  cudaStatus = cudaDeviceSynchronize(); // should be used for the timing.
  cudaStatus = cudaFree(dev_L);
  cudaStatus = cudaFree(d_invQ);
  cudaStatus = cudaFree(d_invQt);
  cusolverStatus = cusolverDnDestroy(handle);
  cudaStatus = cudaDeviceReset();
  
  cudaStatus = cudaFree(Work);
  free(invQ);
  free(invQt);
  free(logdetQ);
}


void gpuPrepareLikelihood::gpu_cholesky() {

  int *d_info, Lwork; //device version of info, worksp.size

//  int info_gpu = 0;

  cudaMalloc((void **) &d_info, sizeof(int));

  // compute workspace size and prepare workspace
  cusolverStatus = cusolverDnSpotrf_bufferSize(handle, uplo, N, d_Q, N, &Lwork);
  cudaStatus = cudaMalloc((void**) &Work, Lwork * sizeof(float));

  cusolverStatus = cusolverDnSpotrf(handle, uplo, N, d_Q, N, Work, Lwork, d_info);

  cusolverStatus = cusolverDnSpotrs(handle, uplo, N, 1, d_Q, N, d_

  cudaMemcpy(b, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(A, d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost);

}



//GPUAdder::GPUAdder (int* array_host_, int length_) {
//  array_host = array_host_;
//  length = length_;
//  int size = length * sizeof(int);
//  cudaError_t err = cudaMalloc((void**) &array_device, size);
//  assert(err == 0);
//  err = cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
//  assert(err == 0);
//}
//
//void GPUAdder::increment() {
//  kernel_add_one<<<64, 64>>>(array_device, length);
//  cudaError_t err = cudaGetLastError();
//  assert(err == 0);
//}
//
//void GPUAdder::retreive() {
//  int size = length * sizeof(int);
//  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
//  cudaError_t err = cudaGetLastError();
//  if(err != 0) { cout << err << endl; assert(0); }
//}
//
//void GPUAdder::retreive_to (int* array_host_, int length_) {
//  assert(length == length_);
//  int size = length * sizeof(int);
//  cudaMemcpy(array_host_, array_device, size, cudaMemcpyDeviceToHost);
//  cudaError_t err = cudaGetLastError();
//  assert(err == 0);
//}
//
//GPUAdder::~GPUAdder() {
//  cudaFree(array_device);
//}
//
//
//
//
