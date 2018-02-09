import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)


cdef extern from "manager.hh":
    cdef cppclass gpuPrepareLikelihood "gpuPrepareLikelihood" :
        float* L
#        np.ndarray[ndim=2, dtype=np.float32_t] L
        int N
#        np.ndarray[ndim=2, dtype=np.float32_t] L
        gpuPrepareLikelihood(np.float32_t* Q, np.float32_t* targets, int N) except+
        void gpu_cholesky();

cdef class pygpulike:
    cdef gpuPrepareLikelihood* g
    cdef int N
#    cdef np.ndarray[ndim=2, dtype=np.float32_t] x

    def __cinit__(self, np.ndarray[ndim=2, dtype=np.float32_t] Q, 
                  np.ndarray[ndim=1, dtype=np.float32_t] targets):
        self.N = len(Q)
        self.g = new gpuPrepareLikelihood(&Q[0, 0], &targets[0], self.N)


    def cholesky(self): 
        self.g.gpu_cholesky()
        L = np.asarray(<np.float32_t[:self.N, :self.N]> self.g.L)
        return(L.T)

        #cdef np.ndarray[np.float32_t, ndim=2] pd_numpy = np.eye(self.N, dtype=np.float32)
        #cdef float *pd = &self.g.L[0]
        #return(pd_numpy)



#        x = np.eye(self.g.N)
#        x = &self.g.L[0, 0]

 #       return(x)



#import cython

# declare the interface to the C code
#cdef extern void c_multiply (double* array, double value, int m, int n)
#
##@cython.boundscheck(False)
##@cython.wraparound(False)
#def multiply(np.ndarray[double, ndim=2, mode="c"] input not None, double value):
#    """
#    multiply (arr, value)
#
#    Takes a numpy arry as input, and multiplies each elemetn by value, in place
#
#    param: array -- a 2-d numpy array of np.float64
#    param: value -- a number that will be multiplied by each element in the array
#
#    """
#    cdef int m, n
#
#    m, n = input.shape[0], input.shape[1]
#
#    c_multiply (&input[0,0], value, m, n)
#
#    return None
#
#def multiply2(np.ndarray[double, ndim=2, mode="c"] input not None, double value):
#    """
#    this method works fine, but is not as future-proof the nupy API might change, etc.
#    """
#    cdef int m, n
#
#    m, n = input.shape[0], input.shape[1]
#
#    c_multiply (<double*> input.data, value, m, n)
#
#    return None
#
