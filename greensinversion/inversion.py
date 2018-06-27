import os
import os.path
import sys
import collections
import copy

import numpy as np
import pylab as pl
import collections
import numbers


try:
    # py2.x
    from urllib import pathname2url
    from urllib import url2pathname
    from urllib import quote
    from urllib import unquote
    from urlparse import urlparse
    from urlparse import urlunparse
    from urlparse import urljoin    
    pass
except ImportError:
    # py3.x
    from urllib.request import pathname2url
    from urllib.request import url2pathname
    from urllib.parse import quote
    from urllib.parse import unquote
    from urllib.parse import urlparse
    from urllib.parse import urlunparse
    from urllib.parse import urljoin
    pass


import greensconvolution

from .sourcevecs import build_flash_source
from .sourcevecs import definereflectors
from .sourcevecs import NaN_in_sourcevecs
from .sourcevecs import build_all_source_vecs
from .regularization import apply_tikhonov_regularization
#from .regularization import apply_regularizer
from .tile_rectangle import build_tiled_rectangle
from .fillholes import fillholes_flat

from .grid import build_gi_grid

try:
    import pyopencl as cl
    pass
except:
    cl=None
    pass

class NotANumberError(Exception):
    pass


def timelimitmatrix(mtx,ny,nx,trange,timelimit):
    nt=trange.shape[0]
    # full array WAS indexed by y,x,and time
    # NOW indexed by time,y,x
    #mtxfull=mtx.reshape(ny,nx,nt,mtx.shape[1]);
    mtxfull=mtx.reshape(nt,ny,nx,mtx.shape[1]);
    timeselect=trange < timelimit
    
    if np.count_nonzero(timeselect) == 0:
        raise ValueError("Did not find any frames suitable for performing inversion of layer. Perhaps you should eliminate your shallowest layer and/or discard fewer frames after the initial flash")

    #mtxreduced=mtxfull[:,:,timeselect,:]
    mtxreduced=mtxfull[timeselect,:,:,:]
    #timeselectmtx=np.ones((ny,nx,1),dtype='d')*timeselect.reshape(1,1,nt)
    timeselectmtx=np.ones((1,ny,nx),dtype='d')*timeselect.reshape(nt,1,1)
    
    newlength=ny*nx*np.count_nonzero(timeselect)

    t_amount = (trange[timeselect][-1]-trange[timeselect][0]) * timeselect.shape[0]*1.0/(timeselect.shape[0]-1)

    # print newlength
    # print mtxreduced.shape
    # print mtx.shape
    # print timeselectmtx.shape

    return (mtxreduced.reshape(newlength,mtx.shape[1]),timeselectmtx.reshape(mtx.shape[0]).astype(np.bool_),t_amount)
    

def generateinversionsteps(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,tstars,ny,nx,trange,depths):
    rowselects=[]
    inversions=[]
    inversionsfull=[]
    inverses=[]
    nresults=[]

    # We go from the shallowest (last) entry to the first (deepest) depth
    prevsourcevecs=flashsourcevecs
    prevcolumnscaling = flashsourcecolumnscaling
    
    # NOTE: reflector source vectors and column scaling are bottom (farthest) up.
    # ... we will construct our matrix from top (nearest) down
    # hence iterating backwards through reflectorsourcevecs and reflectorcolumnscaling
    for cnt in range(len(depths)):
        reflectorcnt=len(depths)-cnt-1
        thesesourcevecs=reflectorsourcevecs[reflectorcnt]
        thesecolumnscaling=reflectorcolumnscaling[reflectorcnt]

        tstar=tstars[reflectorcnt]
        

        if reflectorcnt==0:
            # last entry... nresult covers both prev and current
            nresult=prevsourcevecs.shape[1]+thesesourcevecs.shape[1]
            pass
        else:
            # regular entry... nresult covers only prev
            nresult=prevsourcevecs.shape[1]
            pass
        
        fullmatrix=np.concatenate((prevsourcevecs,thesesourcevecs),axis=1)
        fullcolumnscaling=np.concatenate((prevcolumnscaling,thesecolumnscaling))

        (inversion,rowselect,t_amount)=timelimitmatrix(fullmatrix,ny,nx,trange,tstar*2.0)  # see also greensinversionstep definition of endframe

        # (no longer) scale rows by t_amount to represent that 
        # s*V*x = U'*b  where LHS is an integral 
        # over space (layer by layer) 
        # and RHS is an integral over space and time
        # but we want to normalize the integration per
        # unit length in x and t so as to make 
        # tikhonov parameter invariant. 
        
        # The row_scaling represents the dx*dy*dt 
        # of the time integral, but we (no longer) also need 
        # then to divide by total time of this particular
        # calculation, which is t_amount

        sys.stderr.write("Entering SVD; shape=%s\n" % (str(inversion.shape)))
        (u,s,v)=np.linalg.svd(inversion,full_matrices=False)   # inversion/t_amount
        # note v here is already transposed so u*s*v = inversion
        sys.stderr.write("Finished SVD; shape=%s\n" % (str(inversion.shape)))


        # Scale u and v according to row scaling and column scaling
        # sourcevecs were  multiplied by row_scaling/column_scaling

        # so that  A_scaled (column_scaling*x) = b*row_scaling 
        # or A_scaled = A*row_scaling/column_scaling

        # dividing u by row scaling
        # and multiplying columns of v by column scaling
        # Would make u*s*v the equivalent of the unscaled matrix. 

        # But that is not what we will use u and v for...
        # Instead ut s, and vt form the inverse: 
        # vt * sinv * ut:   where x = vt * sinv * ut * b
        # We want this to apply to an unscaled vector b
        # and give an unscaled result x
        # So we need to scale the columns of ut (rows of u) 
        # by multiplying by row_scaling
        # and scale the rows of vt (columns of v)
        # by dividing by column_scalng

        # note that u_scaled and v_scaled are no longer orthogonal matrices

        u_scaled = u*rowscaling   #/t_amount
        v_scaled = v / fullcolumnscaling[np.newaxis,:]

        #filter_factors = tikhonov_regularization(u, s, v, tikparam)
        #inverse = apply_regularizer(u, s, v, filter_factors)
        #inverse=np.dot(v.T*(1.0/s.reshape(1,s.shape[0])),u.T)

        rowselects.append(rowselect)
        inversions.append(inversion*(fullcolumnscaling[np.newaxis,:]/rowscaling))  # *t_amount
        inversionsfull.append(fullmatrix*(fullcolumnscaling[np.newaxis,:]/rowscaling)) # *t_amount
        inverses.append([u_scaled, s, v_scaled])
        nresults.append(nresult)

        prevsourcevecs=thesesourcevecs
        prevcolumnscaling=thesecolumnscaling
        pass


    return (rowselects,inversions,inversionsfull,inverses,nresults)

r"""
def Get_OpenCL_Context():
    OpenCL_CTX=None

    if cl is None:
        raise ValueError("Exception importing pyopencl (pyopencl is required for OpenCL support)")
    
    # First search for first GPU platform 
    platforms = cl.get_platforms()
    for platform in platforms:
        platform_devices=platform.get_devices()
        has_double_gpu=[bool(device.type & cl.device_type.GPU) and device.preferred_vector_width_double > 0 for device in platform_devices]
        if np.any(has_double_gpu):
            
            OpenCL_CTX = cl.Context(
                #dev_type=cl.device_type.GPU,
                devices=np.array(platform_devices,dtype='O')[np.array(has_double_gpu,dtype=np.bool)])
                # properties=[(cl.context_properties.PLATFORM, platform)]
                
            #self.OpenCL_Platform=platform
            #self.figure_out_version()
            pass
        pass
    
    if OpenCL_CTX is None:
        # fall back to a CPU platform 
        for platform in platforms:
            platform_devices=platform.get_devices()
            has_double_cpu=[bool(device.type & cl.device_type.CPU) and device.preferred_vector_width_double > 0 for device in platform_devices]
            if np.any(has_double_cpu):
                
                OpenCL_CTX = cl.Context(
                    dev_type=cl.device_type.CPU,
                    devices=platform_devices[np.where(has_double_gpu)])
                #OpenCL_Platform=platform
                #self.figure_out_version()
                
                pass
            pass
        pass
    return OpenCL_CTX
"""

performinversionkernel=r"""
typedef unsigned long uint64_t;
typedef long int64_t;

__kernel void dodot(__global const double *matrix,
                    __global const double *vector,
                    __global double *outvec,
                    uint64_t firstvecrow,
                    uint64_t sumsize,
                    uint64_t matrix_ncols)
{
  size_t gid=get_global_id(0); /* gid is which row of the matrix/outvec we are operating on */
  
  size_t start_of_row = matrix_ncols*gid; 
  size_t cnt; 
  double result=0.0;

  for (cnt=0; cnt < sumsize; cnt++) {
    result += matrix[ start_of_row + cnt ]*vector[firstvecrow+cnt];
  }

  outvec[gid]=result;
}


__kernel void dodot_extrafactor(__global const double *matrix,
                    __global const double *vector,
                    __global double *outvec,
                    __global const double *extrafactor,
                    uint64_t firstvecrow,
                    uint64_t sumsize,
                    uint64_t matrix_ncols)
/* multply matrix*vector -> outvec, with an element-by-element multiply of
   outvec by an extra factor */
{
  size_t gid=get_global_id(0); /* gid is which row of the matrix/outvec we are operating on */
  
  size_t start_of_row = matrix_ncols*gid; 
  size_t cnt; 
  double result=0.0;

  for (cnt=0; cnt < sumsize; cnt++) {
    result += matrix[ start_of_row + cnt ]*vector[firstvecrow+cnt];
  }

  outvec[gid]=result*extrafactor[gid];
  //outvec[gid]=matrix[matrix_ncols*gid];

}


__kernel void dodot_subtractfrom(__global const double *matrix,
                                 __global const double *vector,
                                 __global double *outvec,
                                 uint64_t firstvecrow,
                                 uint64_t sumsize,
                                 uint64_t matrix_ncols)
{
  /* dot matrix*vector, subtract from outvec */
  size_t gid=get_global_id(0); /* gid is which row of the matrix/outvec we are operating on */
  
  size_t start_of_row = matrix_ncols*gid; 
  size_t cnt; 
  double result=0.0;

  for (cnt=0; cnt < sumsize; cnt++) {
    result += matrix[ start_of_row + cnt ]*vector[firstvecrow+cnt];
  }

  outvec[gid]-=result; /* WARNING: Non-atomic... make sure nothing else might be messing with this entry!!! */
}


"""



class queuelist(list):

    def __init__(self,simplelist):
        super(queuelist,self).__init__(simplelist)
        pass
    
    def __enter__(self):
        for element in self:
            element.__enter__()
        return self
        
    def __exit__(self,type,value,traceback):
        for element in self:
            element.__exit__(type,value,traceback)
            pass
            
        pass
    pass

def parallelperforminversionsteps(OpenCL_CTX,rowselects,inversions,inversionsfull,inverses,nresults,inputmats, tikparams,GPU_Full_Inverse=False):

    if cl is None:
        raise ValueError("greensinversion.parallelperforminversionsteps: Failed to import PyOpenCL")
    
    
    n_inputs = len(inputmats)
    
    if not isinstance(tikparams,collections.Sequence) and not isinstance(tikparams,np.ndarray):
        # single tikparam... broadcast it over all steps
        tikparams = [ tikparams ]*len(rowselects)
        pass

    tikparams_list = [ copy.copy(tikparams) for inpcnt in range(n_inputs) ]
    
    inversioncoeffs_list=[ [] for inpcnt in range(n_inputs) ]
    errs_list=[ [] for inpcnt in range(n_inputs) ]
    
    
    opencl_dodot = cl.Program(OpenCL_CTX,performinversionkernel).build()
    opencl_dodot_function=opencl_dodot.dodot
    opencl_dodot_function.set_scalar_arg_dtypes([ None, None, None,
                                                  np.uint64, np.uint64, np.uint64 ])

    opencl_dodot_subtractfrom_function=opencl_dodot.dodot_subtractfrom
    opencl_dodot_subtractfrom_function.set_scalar_arg_dtypes([ None, None, None,
                                                               np.uint64, np.uint64, np.uint64 ])

    opencl_dodot_extrafactor_function=opencl_dodot.dodot_extrafactor
    opencl_dodot_extrafactor_function.set_scalar_arg_dtypes([ None, None, None, None,
                                                              np.uint64, np.uint64, np.uint64 ])

    residuals = [ np.array(inputmat.reshape(np.prod(inputmat.shape)),dtype='d',order="C") for inputmat in inputmats ]
    res_buffers = [ cl.Buffer(OpenCL_CTX,cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,hostbuf=residual) for residual in residuals ]


    
    with queuelist([ cl.CommandQueue(OpenCL_CTX,properties=greensconvolution.greensconvolution_calc.OpenCL_GetOutOfOrderDeviceQueueProperties(OpenCL_CTX)) for inpcnt in range(n_inputs) ]) as queue:

        lastiter_wait_events=[()]*n_inputs 
        for cnt in range(len(rowselects)):

            rowselect=rowselects[cnt]

            # assume row selection is contiguous (it is)
            rowselectstart=np.where(rowselect)[0][0]
            rowselectnum=np.where(rowselect)[0][-1]+1-rowselectstart

            
            inversion=inversions[cnt]
            inversionfull=inversionsfull[cnt]
            (ui, si, vi) = inverses[cnt]
            # WARNING: ui, vi have been scaled (see generateinversionsteps()) so they are no longer orthogonal matrices!!!

            # better not to form the inverse just once here,
            # because regularization could be different in each case
            # (should it be???) 

            uitranspose_contiguous = np.ascontiguousarray(ui.T,dtype='d')
            uitranspose_buffer = cl.Buffer(OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=uitranspose_contiguous)

            vitranspose_contiguous = np.ascontiguousarray(vi.T,dtype='d')
            vitranspose_buffer = cl.Buffer(OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=vitranspose_contiguous)
            
            # storage for residual[rowselect] that we extract to determine err
            residualrowselects = [ np.empty(rowselectnum,dtype='d',order="C") for inpcnt in range(n_inputs) ]

            inverse_sis = []
            inverse_si_buffers = []
            
            for inpcnt in range(n_inputs):
                tikparam=tikparams_list[inpcnt][cnt]
                if tikparam is None:
                    # tikhonov regularization disabled
                    #inverse=np.dot(vi.T*(1.0/si.reshape(1,si.shape[0])),ui.T)

                    inverse_sis.append(np.array(1.0/si,dtype='d',order='C'))
                    inverse_si_buffers.append(cl.Buffer(OpenCL_CTX,cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR,hostbuf=inverse_sis[inpcnt]))
                    
                    pass
                else:
                    assert(isinstance(tikparam,numbers.Number))
                    usetikparam=tikparam
                    
                    tikparams_list[inpcnt][cnt]=usetikparam
                    
                    # inverse = apply_tikhonov_regularization(ui, si, vi, usetikparam)
                    d = si/(si**2+(usetikparam)**2)  # Tikhonov parameter interpreted deg K NETD * m^2 of depth / J/m^2 of source intensity


                    inverse_sis.append(np.array(d,dtype='d',order='C'))
                    inverse_si_buffers.append(cl.Buffer(OpenCL_CTX,cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR,hostbuf=inverse_sis[inpcnt]))
                    
                    
                    # inverse = np.dot(v.T*(d.reshape(1,d.shape[0])),u.T)

                    pass
                pass
            
            nresult=nresults[cnt]

            
            # Could probably optimize here a bit by using EnqueueCopyBuffer() rather than COPY_HOST_PTR...
            #inverse_contiguous = np.ascontiguousarray(inverse)
            #inverse_buffer = cl.Buffer(OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=inverse_contiguous)

            # buffer to hold sinverse * utranspose
            sut_buffers = [ cl.Buffer(OpenCL_CTX,cl.mem_flags.READ_WRITE,size=inverse_sis[inpcnt].nbytes) for inpcnt in range(n_inputs) ]
                 
            inversion_contiguous = np.ascontiguousarray(inversion)
            inversion_buffer = cl.Buffer(OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=inversion_contiguous)

            if GPU_Full_Inverse:
                inversionfull_contiguous = np.ascontiguousarray(inversionfull)
                inversionfull_buffer = cl.Buffer(OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=inversionfull_contiguous)
                pass

            
            bestfits = [ np.empty(vitranspose_contiguous.shape[0],dtype='d',order="C") for inpcnt in range(n_inputs) ]
            bestfit_buffers = [ cl.Buffer(OpenCL_CTX,cl.mem_flags.WRITE_ONLY,size=bestfits[inpcnt].nbytes) for inpcnt in range(n_inputs) ]

            reconstructeds = [ np.empty(inversion.shape[0],dtype='d',order="C") for inpcnt in range(n_inputs) ]
            reconstructed_buffers = [ cl.Buffer(OpenCL_CTX,cl.mem_flags.WRITE_ONLY,size=reconstructeds[inpcnt].nbytes) for inpcnt in range(n_inputs) ]
            
            
            
                

            
            #bestfit=np.dot(inverse,residuals[inpcnt][rowselect])

            # multiply utranspose by vector, multiply result by inverse_si
            ut_events= [ opencl_dodot_extrafactor_function(queue[inpcnt],(uitranspose_contiguous.shape[0],),None,
                                                           uitranspose_buffer,
                                                           res_buffers[inpcnt],
                                                           sut_buffers[inpcnt],
                                                           inverse_si_buffers[inpcnt],
                                                           rowselectstart,
                                                           rowselectnum,
                                                           uitranspose_contiguous.shape[1],
                                                           wait_for=lastiter_wait_events[inpcnt])
                         for inpcnt in range(n_inputs) ]
            [ queue[inpcnt].flush() for inpcnt in range(n_inputs) ] # Get computation going
            bestfit_events= [ opencl_dodot_function(queue[inpcnt],(vitranspose_contiguous.shape[0],),None,
                                                    vitranspose_buffer,
                                                    sut_buffers[inpcnt],
                                                    bestfit_buffers[inpcnt],
                                                    0,
                                                    vitranspose_contiguous.shape[1],
                                                    vitranspose_contiguous.shape[1],
                                                    wait_for=(ut_events[inpcnt],))
                              for inpcnt in range(n_inputs) ]
            [ queue[inpcnt].flush() for inpcnt in range(n_inputs) ] # Get computation going
            # get result copying 
            bestfit_enqueue_events=[ cl.enqueue_copy(queue[inpcnt],bestfits[inpcnt],bestfit_buffers[inpcnt],wait_for=(bestfit_events[inpcnt],),is_blocking=False) for inpcnt in range(n_inputs) ]
            
            
            # reconstructed=np.dot(inversion,bestfit)
            reconstructed_events= [ opencl_dodot_function(queue[inpcnt],(inversion.shape[0],),None,
                                                          inversion_buffer,
                                                          bestfit_buffers[inpcnt],
                                                          reconstructed_buffers[inpcnt],
                                                          0,
                                                          inversion.shape[1],
                                                          inversion.shape[1],
                                                          wait_for=(bestfit_events[inpcnt],))
                                    for inpcnt in range(n_inputs) ]
            [ queue[inpcnt].flush() for inpcnt in range(n_inputs) ] # Get computation going
            # get result copying 
            reconstructed_enqueue_events=[ cl.enqueue_copy(queue[inpcnt],reconstructeds[inpcnt],reconstructed_buffers[inpcnt],wait_for=(reconstructed_events[inpcnt],),is_blocking=False) for inpcnt in range(n_inputs) ]
            
            # get residuals[inpcnt][rowselect] also copying so we can look at the residual
            # Is it worth using an OpenCL kernel to subtract the two?
            residualrowselect_enqueue_events=[ cl.enqueue_copy(queue[inpcnt],residualrowselects[inpcnt],res_buffers[inpcnt],wait_for=lastiter_wait_events[inpcnt],device_offset=rowselectstart*residualrowselects[inpcnt].dtype.itemsize,is_blocking=False) for inpcnt in range(n_inputs) ]
            

            
            # observe change in residual 
            #residual=residual-np.dot(inversionfull[:,:nresult],bestfit[:nresult])
            if GPU_Full_Inverse:
                residual_events= [ opencl_dodot_subtractfrom_function(queue[inpcnt],(inversionfull.shape[0],),None,
                                                                      inversionfull_buffer,
                                                                      bestfit_buffers[inpcnt],
                                                                      res_buffers[inpcnt],
                                                                      0,
                                                                      nresult,
                                                                      inversionfull.shape[1],
                                                                      wait_for=(bestfit_events[inpcnt],))
                                   for inpcnt in range(n_inputs) ]
                lastiter_wait_events = [ (residual_event,) for residual_event in residual_events ]  # list of events to wait for at start of next iteration
                pass
            else:
                # Do the full inverse with the CPU, presumably because the GPU doesn't have enough memory to store it
                # Wait for our bestfit data to be copied into place
                [ event.wait() for event in bestfit_enqueue_events ]
                
                if np.isnan(bestfits[inpcnt][:nresult]).any():
                    raise ValueError("Got NAN!")

                residual_update_copy_enqueue_events=[]
                for inpcnt in range(n_inputs):
                    residuals[inpcnt] -= np.dot(inversionfull[:,:nresult],bestfits[inpcnt][:nresult])
                    residual_update_copy_enqueue_events.append((cl.enqueue_copy(queue[inpcnt],res_buffers[inpcnt],residuals[inpcnt],wait_for=(residualrowselect_enqueue_events[inpcnt],),is_blocking=False),))
                    pass

                lastiter_wait_events=[]
                lastiter_wait_events.extend(residual_update_copy_enqueue_events)
                pass


            # Wait for our bestfit data to be copied into place
            [ event.wait() for event in bestfit_enqueue_events ]
            # bestfits numpy arrays are now legitimate

            # print nresults,bestfit.shape
            # print " "
            # print " "
            # inversioncoeffs.extend(list(bestfit[:nresult]))
            [ inversioncoeffs_list[inpcnt].extend(list(bestfits[inpcnt][:nresult])) for inpcnt in range(n_inputs) ]
            

            # wait for reconstruction to become available, so we can evaluate the error
            [ event.wait() for event in reconstructed_enqueue_events ]
            [ event.wait() for event in residualrowselect_enqueue_events ]
            # reconstructed and residualrowselect arrays are now available
            # err=np.linalg.norm(reconstructed-residual[rowselect])
            [ errs_list[inpcnt].append(np.linalg.norm(reconstructeds[inpcnt]-residualrowselects[inpcnt])) for inpcnt in range(n_inputs) ]
            
            
            #print inversion.shape
            #print bestfit.shape
            #print nresult
            #print residual.shape
            
            [ reconstructed_buffer.release() for reconstructed_buffer in reconstructed_buffers ]
            [ bestfit_buffer.release() for bestfit_buffer in bestfit_buffers ]
            if GPU_Full_Inverse:
                inversionfull_buffer.release()
                pass
            inversion_buffer.release()
            #inverse_buffer.release()
            [ sut_buffer.release() for sut_buffer in sut_buffers ]

            [ inverse_si_buffer.release() for inverse_si_buffer in inverse_si_buffers ]
            
            pass
        [ queue[inpcnt].finish() for inpcnt in range(n_inputs) ]
        pass
    
    # convert elements in inversion_coeffs_list to arrays
    inversioncoeffs_list=[ np.array(invcoeffs,dtype='d') for invcoeffs in inversioncoeffs_list ]

        
    return (inversioncoeffs_list,errs_list,tikparams_list)




def performinversionsteps(rowselects,inversions,inversionsfull,inverses,nresults,inputmat, tikparam):
    # tikparam: if None, disable regularization
    #           if a list, use values according to step
    #           if a number, use that value

    inputmat=inputmat.reshape(np.prod(inputmat.shape))
    # assert(inputmat.shape[0]=inverses.shape
    inversioncoeffs=[]
    errs=[]

    tikparams=[]

    residual=inputmat
    
    for cnt in range(len(rowselects)):
        rowselect=rowselects[cnt]
        inversion=inversions[cnt]
        inversionfull=inversionsfull[cnt]
        (ui, si, vi) = inverses[cnt]

        if tikparam is None:
            # tikhonov regularization disabled
            # NOTE: This next line is probably the slow computaton part
            # We should just multiply by the inverse components
            # rather than doing an (expensive) matrix multiply
            #inverse=np.dot(vi.T*(1.0/si.reshape(1,si.shape[0])),ui.T)
            # bestfit=np.dot(inverse,residual[rowselect])

            # Faster: 
            bestfit = np.dot(vi.T,np.dot(ui.T,residual[rowselect])*(1.0/si))
            pass
        else:
            if isinstance(tikparam,collections.Sequence) or isinstance(tikparam,np.ndarray):
                # a list or similar
                usetikparam=tikparam[cnt]
                pass
            else:
                assert(isinstance(tikparam,numbers.Number))
                usetikparam=tikparam
                pass
                
            tikparams.append(usetikparam)
            
            # NOTE: This next line is probably the slow computaton part
            bestfit = apply_tikhonov_regularization(ui, si, vi, usetikparam,residual[rowselect])
            pass
            
        nresult=nresults[cnt]

        reconstructed=np.dot(inversion,bestfit)
        err=np.linalg.norm(reconstructed-residual[rowselect])

        # print nresults,bestfit.shape
        # print " "
        # print " "
        inversioncoeffs.extend(list(bestfit[:nresult]))
        #print inversion.shape
        #print bestfit.shape
        #print nresult
        #print residual.shape
        residual=residual-np.dot(inversionfull[:,:nresult],bestfit[:nresult])
        
        errs.append(err)
        
        pass
    
    return (np.array(inversioncoeffs,dtype='d'),residual,errs,tikparams)

def serialperforminversionsteps(OpenCL_CTX,rowselects,inversions,inversionsfull,inverses,nresults,inputmats, tikparams,GPU_Full_Inverse=False):
    # Does not use OpenCL_CTX or GPU_Full_Inverse
    
    n_inputs=len(inputmats)

    inversioncoeffs_list=[]
    errs_list=[]
    tikparams_list=[]

    for inpcnt in range(n_inputs):

        (inversioncoeffs,residual,errs,tikparams_out)=performinversionsteps(rowselects,inversions,inversionsfull,inverses,nresults,inputmats[inpcnt],tikparams) 
        
        inversioncoeffs_list.append(inversioncoeffs)
        errs_list.append(errs)
        tikparams_list.append(tikparams_out)
        
        pass
    return (inversioncoeffs_list,errs_list,tikparams_list)
    





def generatesinglestepinversion(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,tstars,ny,nx,trange,depths):

    # *** NOTE: Should we use timelimitmatrix to use only a limited range of time for our deepest layer rather than using all frames? 


    # See generateinversionsteps() for more details
    # NOTE: reflector source vectors and column scaling are bottom (farthest) up.
    # ... we will construct our matrix from top (nearest) down
    toconcat=[flashsourcevecs]
    toconcat.extend(reflectorsourcevecs[::-1])

    columnscaling_toconcat=[flashsourcecolumnscaling]
    columnscaling_toconcat.extend(reflectorcolumnscaling[::-1])
    
    inversionall=np.concatenate(toconcat,axis=1)
    columnscalingall=np.concatenate(columnscaling_toconcat)
    
    t_amount=(trange[-1]-trange[0])*trange.shape[0]*1.0/(trange.shape[0]-1)
    

    sys.stderr.write("Entering single step SVD; shape=%s\n" % (str(inversionall.shape)))
    (uiall,siall,viall)=np.linalg.svd(inversionall,full_matrices=False) # inversionall/t_amount
    sys.stderr.write("Completed single step SVD; shape=%s\n" % (str(inversionall.shape)))

    uiall_scaled=uiall*rowscaling  # /t_amount
    viall_scaled=viall / columnscalingall[np.newaxis,:]
    
    # if tikparam is None:
    #    # tikhonov regularization disabled
    #    inverseall=np.dot(viall.T*(1.0/siall.reshape(1,siall.shape[0])),uiall.T)
    #    pass
    #else:
    #    if tikparam==-1:
    #        usetikparam=generate_tikhonov_parameter()
    #filter_factors = tikhonov_regularization(uiall, siall, viall, tikparam)
    #inverseall = apply_regularizer(uiall, siall, viall, filter_factors)
    #
    rowselects=[np.ones(ny*nx*trange.shape[0],dtype=np.bool_)]  # all True
    inversions=[inversionall*(columnscalingall[np.newaxis,:]/rowscaling)]  # t_amount
    inversionsfull=[inversions[0]]
    inverses=[ [uiall_scaled, siall, viall_scaled], ]
    nresults=[inversionall.shape[1]]
    
    return (rowselects,inversions,inversionsfull,inverses,nresults)

def plotabstractinverse(fignum,numplotrows,numplotcols,inversioncoeffs,reflectors,vmin,vmax,y_bnd,x_bnd,num_sources_y,num_sources_x):
    fig=pl.figure(fignum)
    pl.clf()

    inversioncoeffspos=0

    subplots=[]
    images=[]
    for subplotnum in range(1,len(reflectors)+2):
        if subplotnum==1:
            depth=0.0
            ny=num_sources_y
            nx=num_sources_x
            pass
        else:
            (depth,ny,nx)=reflectors[len(reflectors)-subplotnum+1]
            pass
        numampls=ny*nx

        
        subplotcoeffs=inversioncoeffs[inversioncoeffspos:(inversioncoeffspos+numampls)]
        # print subplotcoeffs.shape
        # print ny,nx
        # print inversioncoeffs.shape
        subplot=pl.subplot(numplotrows,numplotcols,subplotnum)
        image=pl.imshow(subplotcoeffs.reshape(ny,nx),vmin=vmin,vmax=vmax,extent=(x_bnd[0]*1.e3,x_bnd[-1]*1.e3,y_bnd[-1]*1.e3,y_bnd[0]*1.e3))
        pl.title('Depth=%f mm' % (depth*1e3))
        pl.grid(True)
        pl.colorbar()

        subplots.append(subplot)
        images.append(image)

        inversioncoeffspos+=numampls
        pass
    return (fig,subplots,images)

def savetiledconcreteinverse(filename,fullinverse,reflectors,yvec,xvec,zthick,zequalszero_on_back_surface=False):
    # Save thermal data as a netcdf (.nc) file
    from netCDF4 import Dataset

    
    rootgrp=Dataset(filename,"w",format="NETCDF4")
    ydim=rootgrp.createDimension("z",fullinverse.shape[0])
    ydim=rootgrp.createDimension("y",fullinverse.shape[1])
    xdim=rootgrp.createDimension("x",fullinverse.shape[2])

    zvals=rootgrp.createVariable("z","f8",("z",))
    for zcnt in range(len(reflectors)):  # reflectors is depth-first
        # first element in each reflectors tuple is z-position, measured from the front surface, positive deep
        if zequalszero_on_back_surface:
            zvals[zcnt]=zthick-reflectors[len(reflectors)-1-zcnt][0]  
            pass
        else:
            zvals[zcnt]=reflectors[len(reflectors)-1-zcnt][0]  
            pass
        pass
    

    yvals=rootgrp.createVariable("y","f8",("y",))
    yvals[:]=yvec
    xvals=rootgrp.createVariable("x","f8",("x",))
    xvals[:]=xvec

    intensityvals=rootgrp.createVariable("sourceintensity","f8",("z","y","x"))
    intensityvals[::]=fullinverse
    
    rootgrp.close()
    
    pass

def buildconcreteinverse(inversioncoeffs,reflectors,ygrid,xgrid,y_bnd,x_bnd,ny,nx,num_sources_y,num_sources_x):

    inversioncoeffspos=0

    res=np.zeros((len(reflectors)+1,ny,nx),dtype='d')
    
    for layercnt in range(len(reflectors)+1):
        if layercnt==0:
            depth=0.0  # front surface: flash_source_vecs
            #reflector_ny=2
            #reflector_nx=2

            # Use flashsourcevecs!
            # !!!*** ... Do we need flashsourcecolumn scaling? (Don't think so)
            #numampls=flashsourcevecs.shape[1]
            numampls=num_sources_y*num_sources_x
            pass
        else: 
            (depth,reflector_ny,reflector_nx)=reflectors[len(reflectors)-layercnt]
            numampls=reflector_ny*reflector_nx
            pass
        
        coeffs=inversioncoeffs[inversioncoeffspos:(inversioncoeffspos+numampls)]
        
        #coeffsshaped=coeffs.reshape(reflector_ny,reflector_nx)
        
        # now expand coeffsshaped into the full (ny,nx) grid
        # need to be consistent with build_reflector_source_vecs

        if layercnt==0:
            # flash_source_vecs naturally overlap
            flash_source=build_flash_source(ygrid,xgrid,y_bnd,x_bnd,num_sources_y,num_sources_x)

            assert(num_sources_y*num_sources_x == numampls)

            # flash_source was (4,ny,nx) (now can be more than 4)  
            # multiply it by source intensity coefficients
            #print coeffs.shape
            #print flash_source.shape
            res[layercnt,:,:]=np.tensordot(coeffs,flash_source,((0,),(0,)))
            pass
        else: 
            # define geometries of each reflector at this depth
            (reflector_widthy,
             reflector_widthx,
             reflector_posy,
             reflector_posx,
             reflector_ygrid,
             reflector_xgrid,
             reflector_bndy,
             reflector_bndx)=definereflectors(y_bnd,x_bnd,reflector_ny,reflector_nx)
            
            # Iterate over which reflector...
            for refl_yidx in range(reflector_ny):
                for refl_xidx in range(reflector_nx):
                    coeffidx=reflector_nx*refl_yidx + refl_xidx

                    # Determine range of xy points corresponding to this reflector
                    refl_xygrid=((ygrid[:,:] >= reflector_bndy[refl_yidx]) & 
                                 (ygrid[:,:] < reflector_bndy[refl_yidx+1]) & 
                                 (xgrid[:,:] >= reflector_bndx[refl_xidx]) & 
                                 (xgrid[:,:] < reflector_bndx[refl_xidx+1]))
                    #print(res.shape)
                    #print(refl_xygrid.shape)
                    res[layercnt,refl_xygrid]=coeffs[coeffidx]
                    pass # end loop reflector_nx
                pass # end loop reflector_ny
            pass # end layercnt != 0
        
        inversioncoeffspos+=numampls

        pass # end loop layercnt
    return res   # (len(reflectors)+1,ny,nx)... first layer is surface



def plotconcreteinverse(fignum,numplotrows,numplotcols,saturation_map,concreteinverse,reflectors,vmin,vmax,y_bnd,x_bnd,num_sources_y,num_sources_x):
    fig=pl.figure(fignum)
    pl.clf()

    subplots=[]
    images=[]
    for subplotnum in range(1,concreteinverse.shape[0]+2):
        if subplotnum <= 2:
            depth=0.0
            ny=num_sources_y
            nx=num_sources_x
            pass
        else:
            (depth,ny,nx)=reflectors[len(reflectors)-(subplotnum-2)]
            pass

        subplot=pl.subplot(numplotrows,numplotcols,subplotnum)
        if subplotnum==1:
            # Show saturation map
            image=pl.imshow(saturation_map,extent=(x_bnd[0]*1.e3,x_bnd[-1]*1.e3,y_bnd[-1]*1.e3,y_bnd[0]*1.e3))
            pl.title('Saturation map')
            pass
        else:
            image=pl.imshow(concreteinverse[subplotnum-2,:,:]/1.e3,vmin=vmin/1.e3,vmax=vmax/1.e3,extent=(x_bnd[0]*1.e3,x_bnd[-1]*1.e3,y_bnd[-1]*1.e3,y_bnd[0]*1.e3))
            pl.title('Depth=%.2f mm' % (depth*1e3))
            pass
        pl.grid(True)
        pl.colorbar()

        subplots.append(subplot)
        images.append(image)
        pass
    return (fig,subplots,images)



def plotconcreteinversemovie(startfignum,outdirhref,outfilenametemplate,saturation_map,concreteinverse,reflectors,vmin,vmax,y_bnd,x_bnd,num_sources_y,num_sources_x,**savefigkwargs):

    if outdirhref is not None: 
        from limatix import dc_value
        pass
    
    if outdirhref is not None and not os.path.exists(outdirhref.getpath()):
        os.mkdir(outdirhref.getpath())
        pass

    plots=[]
    images=[]
    plothrefs=[]
    depths=[]
    for plotnum in range(concreteinverse.shape[0]+1):
        fig=pl.figure(startfignum+plotnum)
        pl.clf()
        
        if plotnum <= 1:
            depth=0.0
            ny=num_sources_y
            nx=num_sources_x
            pass
        else:
            (depth,ny,nx)=reflectors[len(reflectors)-(plotnum-1)]
            pass

        #subplot=pl.subplot(numplotrows,numplotcols,subplotnum)
        if plotnum==0:
            # Show saturation map
            image=pl.imshow(saturation_map,extent=(x_bnd[0]*1.e3,x_bnd[-1]*1.e3,y_bnd[-1]*1.e3,y_bnd[0]*1.e3))
            pl.title('Saturation map')
            pass
        else:
            image=pl.imshow(concreteinverse[plotnum-1,:,:]/1.e3,vmin=vmin/1.e3,vmax=vmax/1.e3,extent=(x_bnd[0]*1.e3,x_bnd[-1]*1.e3,y_bnd[-1]*1.e3,y_bnd[0]*1.e3))
            pl.title('Depth=%.2f mm' % (depth*1e3))
            pass
        pl.grid(True)
        pl.colorbar()

        pl.xlabel('Position (mm)')
        pl.ylabel('Position (mm)')

        if outdirhref is not None:
            outfilename=outfilenametemplate % (depth*1e3)
            outfilehref=dc_value.hrefvalue(quote(outfilename),contexthref=outdirhref)
            outfilepath=outfilehref.getpath()
            pl.savefig(outfilepath,**savefigkwargs)

            plothrefs.append(outfilehref)
            pass
            
        plots.append(fig)
        images.append(image)
        depths.append(depth)
        pass
    return (startfignum+concreteinverse.shape[0],plots,images,plothrefs,depths)


def define_curved_inversion(gi_params,gi_grid,obj,curvmat_tile,stepsizemat_tile,curvmat_hires,stepsizemat_hires,curvmat_sizeu,curvmat_sizev,num_sources_y,num_sources_x):
    (rho,c,alphaz,alphaxy,dy,dx,maxy,maxx,t0,dt,nt,reflectors,
     trange,greensconvolution_params) = gi_params
    (ny,nx,y,x,ygrid,xgrid,y_bnd,x_bnd) = gi_grid

    kz=alphaz*rho*c
    kx=alphaxy*rho*c
    ky=alphaxy*rho*c

    #eval_linelength_avgcurvature = lambda u1,v1,u2,v2: obj.implpart.surfaces[0].intrinsicparameterization.linelength_avgcurvature(obj.implpart.surfaces[0],dx,dy,u1,v1,u2,v2)
    # eval_linelength_avgcurvature = lambda u1,v1,u2,v2: obj.implpart.surfaces[0].intrinsicparameterization.linelength_avgcurvature_meshbased(obj.implpart.surfaces[0],curvmat_hires,stepsizemat_hires,dx,dy,u1,v1,u2,v2)
    eval_linelength_avgcurvature_mirroredbox = lambda boxu1,boxv1,boxu2,boxv2,u1,v1,u2,v2: obj.implpart.surfaces[0].intrinsicparameterization.linelength_avgcurvature_mirroredbox_meshbased(obj.implpart.surfaces[0],curvmat_hires,stepsizemat_hires,obj.implpart.surfaces[0].intrinsicparameterization.lowerleft_meaningfulunits[0],obj.implpart.surfaces[0].intrinsicparameterization.lowerleft_meaningfulunits[1],curvmat_sizeu*1.0/curvmat_hires.shape[1],curvmat_sizev*1.0/curvmat_hires.shape[0],boxu1,boxv1,boxu2,boxv2,dx,dy,u1,v1,u2,v2)


    print("Building curved sourcevecs")
    (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions)=build_all_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,dt,trange,reflectors,gc_kernel="opencl_interpolator_curved",eval_linelength_avgcurvature_mirroredbox=eval_linelength_avgcurvature_mirroredbox,curvmat_tile=curvmat_tile,stepsizemat_tile=stepsizemat_tile,num_sources_y=num_sources_y,num_sources_x=num_sources_x)

    if NaN_in_sourcevecs([ flashsourcevecs ]):
        raise NotANumberError("NaN found in flashsourcevecs")

    if NaN_in_sourcevecs(reflectorsourcevecs):
        raise NotANumberError("NaN found in reflectorsourcevecs")
        
    if singlestep:
        print("Generating single-step curved inversion")
        (rowselects,inversions,inversionsfull,inverses,nresults)=generatesinglestepinversion(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,tstars,ny,nx,trange,depths)
        pass
    else:
        print("Generating curved inversion steps")
        (rowselects,inversions,inversionsfull,inverses,nresults)=generateinversionsteps(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,tstars,ny,nx,trange,depths)
        pass
        
    
    return (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions,rowselects,inversions,inversionsfull,inverses,nresults)

def define_flat_inversion(gi_params,gi_grid,num_sources_y,num_sources_x,singlestep=False):
    (rho,c,alphaz,alphaxy,dy,dx,maxy,maxx,t0,dt,nt,reflectors,
     trange,greensconvolution_params) = gi_params
    (ny,nx,y,x,ygrid,xgrid,y_bnd,x_bnd) = gi_grid

    kz=alphaz*rho*c
    kx=alphaxy*rho*c
    ky=alphaxy*rho*c


    print("Building flat sourcevecs")

    (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions)=build_all_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,dt,trange,reflectors,num_sources_y=num_sources_y,num_sources_x=num_sources_x)
    
    if singlestep:
        print("Generating single-step flat inversion")
    
        (rowselects,inversions,inversionsfull,inverses,nresults)=generatesinglestepinversion(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,tstars,ny,nx,trange,depths)
        pass
    else:

        print("Generating flat inversion steps")

        (rowselects,inversions,inversionsfull,inverses,nresults)=generateinversionsteps(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,tstars,ny,nx,trange,depths)
        pass


    return (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions,rowselects,inversions,inversionsfull,inverses,nresults)

def saturationcheck(thermal_data,startframe,sat_threshold=0.9):
    """ Determine the fraction of thermal_data that is saturated
    on or after startframe (0 based). It is assumed the highest
    temperature recorded in thermal_data for a particular pixel 
    is the saturation value and that the thermal data has already
    been background-subtracted. 
    
    A pixel is defined as saturated is defined as exceeding sat_threshold*max_for_that_pixel(thermal_data)
   
    Returns a tuple containing a number between 0 and 1 representing 
    the fraction of valid pixels (not identically 0, not infinite, not NaN) 
    that are saturated, followed by a saturation map 
"""
    saturation_levels = np.max(thermal_data,axis=0)

    saturated = np.sum(thermal_data[startframe:,:,:] > sat_threshold*saturation_levels[np.newaxis,:,:],axis=0) > 0

    valid = np.isfinite(saturation_levels) & (saturation_levels != 0.0)
    
    fraction_saturated = np.count_nonzero(saturated)*1.0/np.count_nonzero(valid)

    return (fraction_saturated,(saturated & valid))

def num_sources(y,x,y_bnd,x_bnd,source_approx_dy,source_approx_dx):
    # num_sources_y should approximately match
    # (y_bnd[-1]-y_bnd[0])/source_approx_dy, BUT
    #   ... it should be an integer AND
    #   ... not larger than y.shape[0] i.e. at least 1 pixel/source
    num_sources_y = int(round((y_bnd[-1]-y_bnd[0])/source_approx_dy))
    if num_sources_y > y.shape[0]:
        num_sources_y=y.shape[0]
        pass

    # num_sources_x should approximately match
    # (x_bnd[-1]-x_bnd[0])/source_approx_dx, BUT
    #   ... it should be an integer AND
    #   ... not larger than x.shape[0] i.e. at least 1 pixel/source
    num_sources_x = int(round((x_bnd[-1]-x_bnd[0])/source_approx_dx))
    if num_sources_x > x.shape[0]:
        num_sources_x=x.shape[0]
        pass

    return (num_sources_y,num_sources_x)


def setupinversionprob(rho,c,alphaz,alphaxy,dy,dx,maxy,maxx,t0,dt,nt,reflectors,source_approx_dy=None,source_approx_dx=None,singlestep=False):
    
    kx=alphaxy*rho*c
    ky=alphaxy*rho*c
    kz=alphaz*rho*c
        
    trange=t0+np.arange(nt,dtype='d')*dt
    
    greensconvolution_params=greensconvolution.greensconvolution_calc.read_greensconvolution()

    gi_params=(rho,c,alphaz,alphaxy,dy,dx,maxy,maxx,t0,dt,nt,reflectors,
     trange,greensconvolution_params)
    
    gi_grid = build_gi_grid(dy,maxy,
                            dx,maxx)
    (ny,nx,
     y,x,
     ygrid,xgrid,
     y_bnd,x_bnd) = gi_grid

    num_sources_y=2
    num_sources_x=2

    if source_approx_dy is not None or source_approx_dx is not None:
        (num_sources_y,num_sources_x) = num_sources(y,x,y_bnd,x_bnd,source_approx_dy,source_approx_dx)
        pass

    
        #print("Building source vecs")
    
    (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions,rowselects,inversions,inversionsfull,inverses,nresults)=define_flat_inversion(gi_params,gi_grid,num_sources_y,num_sources_x,singlestep=singlestep)
    
    
    
    inversionprob=(kx,ky,kz,
                   ny,nx,
                   y,x,
                   ygrid,xgrid,
                   y_bnd,x_bnd,
                   num_sources_y,num_sources_x,
                   trange,
                   rowscaling,
                   flashsourcecolumnscaling,flashsourcevecs,
                   reflectorcolumnscaling,reflectorsourcevecs,
                   depths,tstars,
                   conditions,prevconditions,prevscaledconditions,
                   rowselects,inversions,inversionsfull,inverses,nresults)
    return inversionprob


def perform_flat_inversion(rho,c,alphaz,alphaxy,y0,x0,dy,dx,tile_size_y,tile_size_x,xydownsample,reflectors,source_approx_dy,source_approx_dx,tikparam,data_t0,dt,flashframe,flashtime,frames_to_discard,frames_to_discard_prior,data,singlestep,parallelevaluate,OpenCL_Device_Type,OpenCL_Device_Name,numplotrows,numplotcols,plot_min_power_per_area,plot_max_power_per_area,nextfignum):

    # Perform background subtraction
    bkgnd_frames = flashframe-frames_to_discard_prior
    background=np.mean(data[:bkgnd_frames,:,:],axis=0)
    startframe = flashframe + frames_to_discard-1
    
    data_timebase=data_t0+np.arange(data.shape[0],dtype='d')*dt
    
    t0=data_timebase[startframe] - flashtime
    
    diff = data-background[np.newaxis,:,:]
    
    (saturation_fraction,saturation_map)=saturationcheck(diff,startframe)
    
    if saturation_fraction > .2: 
        raise ValueError("TWIRAW_greensinversion: ERROR: %.1f%% of pixels are saturated at least once beyond start frame!" % (saturation_fraction*100.0))
    if saturation_fraction > .02:
        sys.stderr.write("TWIRAW_greensinversion: WARNING: %.1f%% of pixels are saturated at least once beyond start frame!\n" % (saturation_fraction*100.0))
        pass
    
    deepest_tstar = reflectors[0][0]**2.0/(np.pi*alphaz)
    endframe = np.argmin(np.abs(data_timebase-data_timebase[flashframe]-flashtime-deepest_tstar*2.0))   # see also generateinversionsteps() call to timelimitmatrix()

    nt=data_timebase[startframe:endframe].shape[0]
    
    inversionprob = setupinversionprob(rho,c,alphaz,alphaxy,dy,dx,tile_size_y,tile_size_x,t0,dt,nt,reflectors,source_approx_dy=source_approx_dy,source_approx_dx=source_approx_dx,singlestep=singlestep)
    

    (kx,ky,kz,
     ny,nx,
     y,x,
     ygrid,xgrid,
     y_bnd,x_bnd,
     num_sources_y,num_sources_x,
     trange,
     rowscaling,
     flashsourcecolumnscaling,flashsourcevecs,
     reflectorcolumnscaling,reflectorsourcevecs,
     depths,tstars,
     conditions,prevconditions,prevscaledconditions,
     rowselects,inversions,inversionsfull,inverses,nresults) = inversionprob

    (minyminx_corners,yranges,xranges,contributionprofiles)=build_tiled_rectangle(ny,nx,dy,dx,reflectors,diff,xydownsample)


    
    inputmats = [ diff[startframe:endframe,(yidx*xydownsample):((yidx+ny)*xydownsample):xydownsample,(xidx*xydownsample):((xidx+nx)*xydownsample):xydownsample] for (yidx,xidx) in minyminx_corners ]  
    print("Filling holes...")
    inputmats_holesfilled = [ fillholes_flat(inputmat) for inputmat in inputmats ]
    print("Done filling holes.")


    if parallelevaluate:
        inversionevalfunc=parallelperforminversionsteps
        OpenCL_CTX=greensconvolution_params.get_opencl_context()   #greensinversion.inversion.Get_OpenCL_Context()
        pass
    else:
        inversionevalfunc=serialperforminversionsteps
        OpenCL_CTX=None
        pass
        
    if nextfignum is not None:
        from matplotlib import pyplot as pl
        # tikparam diagnostic plots
        pl.figure(nextfignum)
        pl.clf()
        for inversioncnt in range(len(inversions)):
            pl.plot(inverses[inversioncnt][1])
            pass
        pl.xlabel('Singular value index')
        pl.ylabel('Magnitude')
        nextfignum+=1
        pass

        
    # evaluate inversion
    (inversioncoeffs_list,errs_list,tikparams_list) = inversionevalfunc(OpenCL_CTX,
                                                                        rowselects,
                                                                        inversions,
                                                                        inversionsfull,
                                                                        inverses,
                                                                        nresults,
                                                                        inputmats_holesfilled,
                                                                        tikparam)
    
    # Generate concrete representation of inversion
    
    
    fullinverse=np.zeros((len(reflectors)+1,diff.shape[1]//xydownsample,diff.shape[2]//xydownsample),dtype='d')
    fullinverse_x_bnd=x0-dx*xydownsample/2.0 + np.arange(diff.shape[2]//xydownsample+1,dtype='d')*dx*xydownsample
    fullinverse_y_bnd=y0-dy*xydownsample/2.0 + np.arange(diff.shape[1]//xydownsample+1,dtype='d')*dy*xydownsample
    
    for tile_idx in range(len(minyminx_corners)):
        (yidx,xidx)=minyminx_corners[tile_idx]
        
        fullinverse[:,yidx:(yidx+ny),xidx:(xidx+nx)] += buildconcreteinverse(inversioncoeffs_list[tile_idx],reflectors,ygrid,xgrid,y_bnd,x_bnd,ny,nx,num_sources_y,num_sources_x)*contributionprofiles[tile_idx]
        pass
        
        
    # Plot concrete inverse as a bunch of subplots
    if nextfignum is not None:
        (fig,subplots,images)=plotconcreteinverse(nextfignum,numplotrows,numplotcols,saturation_map,fullinverse,reflectors,plot_min_power_per_area,plot_max_power_per_area,fullinverse_y_bnd,fullinverse_x_bnd,num_sources_y,num_sources_x)
        nextfignum+=1
        pass

    # Plot separate plots with concrete inverse
    if nextfignum is not None:
        (nextfignum,plots,images,plothrefs,depths) = plotconcreteinversemovie(nextfignum,None,None,saturation_map,fullinverse,reflectors,plot_min_power_per_area,plot_max_power_per_area,fullinverse_y_bnd,fullinverse_x_bnd,num_sources_y,num_sources_x,dpi=300)
        pass

    inversion_info=(minyminx_corners,
                    yranges,
                    xranges,
                    contributionprofiles,
                    inversioncoeffs_list,
                    errs_list,
                    tikparams_list,
                    fullinverse_y_bnd,
                    fullinverse_x_bnd)

    return (inversionprob,saturation_map,
            inversion_info,
            fullinverse,
            nextfignum)

    
