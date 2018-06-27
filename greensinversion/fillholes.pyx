import numpy as np
cimport numpy as np
cimport cython
cimport libc.math

from libc.stdint cimport int32_t,uint32_t,int64_t,uint64_t,int8_t,uint8_t


cdef extern from "fillholes_c.h":
  void fillholes_flat_c(float *outputmat,uint8_t *maskarray,size_t nt,size_t ny,size_t nx)
  pass


def fillholes_flat(inputmat):
    """ Replace any NaN's in inputmat with reasonable numbers.
        This accommodates inversion near irregular edges that don't
        line up with tile boundaries. 
    
    inputmat is an nt by ny by nx matrix of floats, stored C order
    """

    cdef np.ndarray[np.float32_t,ndim=3,mode="c"] outputmat=np.ascontiguousarray(inputmat,dtype=np.float32)
    (nt,ny,nx)=inputmat.shape

    cdef np.ndarray[np.uint8_t,ndim=2,mode="c"] maskarray=np.ascontiguousarray(np.isfinite(inputmat[0,:,:]),dtype=np.uint8)

    fillholes_flat_c(<np.float32_t *>outputmat.data,<np.uint8_t *>maskarray.data,nt,ny,nx)
    
    return outputmat
