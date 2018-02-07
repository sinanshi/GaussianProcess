import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "manager.hh":
    cdef cppclass C_GPUAdder "GPUAdder":
        C_GPUAdder(np.int32_t*, int)
        void increment()
        void retreive()
        void retreive_to(np.int32_t*, int)

cdef class GPUAdder:
    cdef C_GPUAdder* g
    cdef int dim1

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.int32_t] arr):
        self.dim1 = len(arr)
        self.g = new C_GPUAdder(&arr[0], self.dim1)

    def increment(self):
        self.g.increment()

    def retreive_inplace(self):
        self.g.retreive()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] a = np.zeros(self.dim1, dtype=np.int32)
        self.g.retreive_to(&a[0], self.dim1)
        return a


cdef extern from "manager.hh":
    cdef cppclass GPULike :
        GPULike(np.float64_t*) except+
        void display()
        void cholesky(np.float32_t *A, np.float32_t *b, int N);

cdef class pygpulike:
    cdef gpuPrepareLikelihood* g

    def __cinit__(self, np.ndarray[ndim=2, dtype=np.float32_t] Q, 
                  np.ndarray(ndim=1, dtype=np.float32_t) targets):
        self.g = new gpuPrepareLikelihood(&arr[0, 0], &targets[0])


    def cholesky(self, np.ndarray[ndim=2, dtype=np.float32_t] A, 
                 np.ndarray[ndim=1, dtype=np.float32_t] b):
        N = len(A)
        self.g.cholesky(&A[0, 0], &b[0], N)
        return A


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
