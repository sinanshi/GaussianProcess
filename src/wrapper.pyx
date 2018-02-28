import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)


cdef extern from "manager.hh":
    cdef cppclass gpuPrepareLikelihood "gpuPrepareLikelihood" :
        double* invQ
        double* invQt
        double logdetQ
        int N
        gpuPrepareLikelihood(np.float64_t* Q, np.float64_t* targets, int N) except+
        void gpu_cholesky();

cdef class pygpulike:
    cdef gpuPrepareLikelihood* g
    cdef int N

    def __cinit__(self, np.ndarray[ndim=2, dtype=np.float64_t] Q, 
                  np.ndarray[ndim=1, dtype=np.float64_t] targets):
        self.N = len(Q)
        self.g = new gpuPrepareLikelihood(&Q[0, 0], &targets[0], self.N)


    def cholesky(self): 
        self.g.gpu_cholesky()
        invQ = np.asarray(<np.float64_t[:self.N, :self.N]> self.g.invQ)
        invQt = np.asarray(<np.float64_t[:self.N]> self.g.invQt)
        return([invQ.T, invQt, self.g.logdetQ])
