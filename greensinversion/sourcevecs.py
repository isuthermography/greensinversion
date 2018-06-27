import sys

from distutils.version import StrictVersion  # for checking Numpy version

import numpy as np
import numpy.linalg
import scipy
import scipy.special

#  These routine generate synthetic source vectors, either
# for the top surface (build_flash_source_vecs()) broken
# into four rectangles, or for
# an array of buried reflectors at a particular depth
# (build_reflector_source_vecs)
#
#   Ax=b
#   x represents amplitudes of sources (flash, reflectors, and interference)
#   b represents measures thermal image sequence at surface
#
#   so each column in A represents a source.
#   each row represents a detector pixel at a particular time

# The second index of the source vectors represents which source vector
# ... for build_reflector_source_vecs(), i.e. which column in A
# The first index represents which detector pixel at which time
#
# The first index has range ny*nx*nt, and can be reshaped into
# (ny,nx,nt).
# The second index from build_reflector_source_vecs()
# has range reflector_ny*reflector_nx and can
# be reshaped into (reflector_ny,reflector_nx)


from greensconvolution.greensconvolution_fast import greensconvolution_integrate_anisotropic,greensconvolution_image_sources,greensconvolution_greensfcn_curved
from greensconvolution.greensconvolution_calc import OpenCL_GetOutOfOrderDeviceQueueProperties

try:
    import pyopencl as cl
    pass
except:
    cl=None
    pass


def mtxnorm_over_axis_0(mtx):
    # workaround version for numpy < 1.8
    numcols=mtx.shape[1]
    result=np.zeros(numcols,dtype=mtx.dtype)
    for cnt in range(numcols):
        result[cnt]=np.linalg.norm(mtx[:,cnt])
        pass
        
    return result

def scaledcondition(mtx):
    # Normalize each column, then determine the condition number
    if StrictVersion(np.__version__) >= StrictVersion('1.8.0'):
        # only numpy >= 1.8.0 has axis parameter to mtxnorm
        mtxnorm=np.linalg.norm(mtx,axis=0)
        pass
    else: 
        mtxnorm=mtxnorm_over_axis_0(mtx)
        pass

    scaledmtx=mtx*(1.0/mtxnorm.reshape(1,mtx.shape[1]))
    s=np.linalg.svd(scaledmtx,full_matrices=False,compute_uv=False)
    return max(s)/min(s)


def build_flash_source(ygrid,xgrid,y_bnd,x_bnd,num_sources_y=2,num_sources_x=2):
    ysize=y_bnd[-1]-y_bnd[0]
    xsize=x_bnd[-1]-x_bnd[0]
    ny=ygrid.shape[0]
    nx=ygrid.shape[1]

    y=ygrid[:,0]
    x=xgrid[0,:]

    sourcehalfwidthy = ysize/(num_sources_y-1)
    sourcehalfwidthx = xsize/(num_sources_x-1)

    flash_source = np.zeros((num_sources_y*num_sources_x,ny,nx),dtype='d')

    for src_ycnt in range(num_sources_y):
        src_centery = y_bnd[0] + src_ycnt * ysize / (num_sources_y-1)
        src_regiony = (y >= src_centery-sourcehalfwidthy) & (y <= src_centery+sourcehalfwidthy)
        for src_xcnt in range(num_sources_x):
            src_centerx = x_bnd[0] + src_xcnt * xsize / (num_sources_x-1)
            src_regionx = (x >= src_centerx-sourcehalfwidthx) & (x <= src_centerx+sourcehalfwidthx)

            sourcevecidx=num_sources_x*src_ycnt + src_xcnt

            ygridregion=ygrid[src_regiony,:][:,src_regionx]
            xgridregion=xgrid[src_regiony,:][:,src_regionx]

            source_intensity = np.cos((xgridregion-src_centerx)*(np.pi/2.0/(sourcehalfwidthx)))**2.0  * np.cos((ygridregion-src_centery)*(np.pi/2.0/(sourcehalfwidthy)))**2.0

            xygridregion = (ygrid >= src_centery-sourcehalfwidthy) & (ygrid <= src_centery+sourcehalfwidthy) & (xgrid >= src_centerx-sourcehalfwidthx) & (xgrid <= src_centerx+sourcehalfwidthx)
            
            flash_source[sourcevecidx,xygridregion] = source_intensity.ravel()
            pass
        pass

    return flash_source

def build_flash_source_curved(greensconvolution_params,y,x,ygrid,xgrid,y_bnd,x_bnd,stepsizemat_tile,rho,c,kz,ky,kx,trange,eval_linelength_avgcurvature_mirroredbox,num_sources_y,num_sources_x):
    ysize=y_bnd[-1]-y_bnd[0]
    xsize=x_bnd[-1]-x_bnd[0]
    ny=ygrid.shape[0]
    nx=ygrid.shape[1]

    sourcehalfwidthy = ysize/(num_sources_y-1)
    sourcehalfwidthx = xsize/(num_sources_x-1)
    nt=trange.shape[0]
    t_reshape=trange.reshape(nt,1,1,1)
    
    alphaz=kz/(rho*c)
    alphay=ky/(rho*c)
    alphax=kx/(rho*c)

    next_closure=None

    greensconvolution_params.get_opencl_context()
    opencl_queue=cl.CommandQueue(greensconvolution_params.OpenCL_CTX,properties=OpenCL_GetOutOfOrderDeviceQueueProperties(greensconvolution_params.OpenCL_CTX))

    
    # ***!!! We don't really handle kx != ky properly
    # (i.e. in-plane anisotropy.) We would need to 
    # explicitly consider what happens along the axis of 
    # propagation, and how that relates to the axis of
    # curvature !!!***
    alphaxy=np.sqrt(alphax*alphay) 
    kxy=np.sqrt(kx*ky) 

    
    flashsourcevecs=np.zeros((ny*nx*nt,num_sources_y*num_sources_x),dtype='d',order='F')
    
    assert(stepsizemat_tile.shape[:2] == ygrid.shape)

    # superposition of cos^2 functions that adds to 1 everywhere
    centerpoint_y = (y_bnd[-1]+y_bnd[0])/2.0
    centerpoint_x = (x_bnd[-1]+x_bnd[0])/2.0


    #sys.modules["__main__"].__dict__.update(globals())
    #sys.modules["__main__"].__dict__.update(locals())
    #raise ValueError("FOO!") # !!!

    for src_ycnt in range(num_sources_y):
        src_centery = y_bnd[0] + src_ycnt * ysize / (num_sources_y-1)
        src_regiony = (y >= src_centery-sourcehalfwidthy) & (y <= src_centery+sourcehalfwidthy)
        for src_xcnt in range(num_sources_x):
            src_centerx = x_bnd[0] + src_xcnt * xsize / (num_sources_x-1)
            src_regionx = (x >= src_centerx-sourcehalfwidthx) & (x <= src_centerx+sourcehalfwidthx)

            sourcevecidx=num_sources_x*src_ycnt + src_xcnt

            # source intensity within src_regiony,src_regionx as a function of position for this source

            ygridregion=ygrid[src_regiony,:][:,src_regionx]
            xgridregion=xgrid[src_regiony,:][:,src_regionx]
            stepsizeregion=stepsizemat_tile[src_regiony,:,:][:,src_regionx,:]

            # source intensity = weighting factor (product of cos^2's) 
            # times actual dx * actual dy  
            source_intensity = np.cos((xgridregion-src_centerx)*(np.pi/2.0/(sourcehalfwidthx)))**2.0  * np.cos((ygridregion-src_centery)*(np.pi/2.0/(sourcehalfwidthy)))**2.0  * stepsizeregion[:,:,0] * stepsizeregion[:,:,1]


            off_y = ygridregion-y_bnd[0]
            off_x = xgridregion-x_bnd[0]
            endoff_y = y_bnd[-1]-ygridregion
            endoff_x = x_bnd[-1]-xgridregion

            # Include image sources mirrored across borders
            # by creating a 3x3 tile array centered over our nominal
            # tile
            mirror_els_y = np.array((y_bnd[0] - off_y,
                                     y_bnd[0] - off_y,
                                     y_bnd[0] - off_y,
                                     y_bnd[0] + off_y,
                                     y_bnd[0] + off_y,
                                     y_bnd[0] + off_y,
                                     y_bnd[-1] + endoff_y,
                                     y_bnd[-1] + endoff_y,
                                     y_bnd[-1] + endoff_y),dtype='d')
            mirror_els_x = np.array((x_bnd[0] - off_x,
                                     x_bnd[0] + off_x,
                                     x_bnd[-1] + endoff_x,
                                     x_bnd[0] - off_x,
                                     x_bnd[0] + off_x,
                                     x_bnd[-1] + endoff_x,
                                     x_bnd[0] - off_x,
                                     x_bnd[0] + off_x,
                                     x_bnd[-1] + endoff_x),dtype='d')
            elmask=np.ones((9,mirror_els_x.shape[1],mirror_els_x.shape[2]),dtype=np.bool)
            # Filter out sources on the opposite side
            elmask[6,(y[src_regiony] < centerpoint_y),:] = False
            elmask[7,(y[src_regiony] < centerpoint_y),:] = False
            elmask[8,(y[src_regiony] < centerpoint_y),:] = False
            elmask[0,(y[src_regiony] >= centerpoint_y),:] = False
            elmask[1,(y[src_regiony] >= centerpoint_y),:] = False
            elmask[2,(y[src_regiony] >= centerpoint_y),:] = False
            elmask[2,:,(x[src_regionx] < centerpoint_x)] = False
            elmask[5,:,(x[src_regionx] < centerpoint_x)] = False
            elmask[8,:,(x[src_regionx] < centerpoint_x)] = False
            elmask[0,:,(x[src_regionx] >= centerpoint_x)] = False
            elmask[3,:,(x[src_regionx] >= centerpoint_x)] = False
            elmask[6,:,(x[src_regionx] >= centerpoint_x)] = False

            # collapse mirror_els_x and mirror_els_y according to elmask
            mirror_els_x_collapse = mirror_els_x[elmask]
            mirror_els_y_collapse = mirror_els_y[elmask]
            source_intensity_collapse=(np.ones((9,1,1),dtype='d')*source_intensity)[elmask]
            step_size_x_collapse=(np.ones((9,1,1),dtype='d')*stepsizeregion[:,:,0])[elmask]
            step_size_y_collapse=(np.ones((9,1,1),dtype='d')*stepsizeregion[:,:,1])[elmask]
            mirror_num_els=mirror_els_x_collapse.shape[0]


            (linelength,avgcurvature,avgcrosscurvature,theta) = (   # theta is 0 for the line at (u2,v2) being in the +u direction, pi/2 for the line being in the +v direction
                eval_linelength_avgcurvature_mirroredbox(
                    # box corners:
                    x_bnd[0],y_bnd[0],
                    x_bnd[-1],y_bnd[-1],
                    xgrid[:,:].reshape(1,ny,nx,1),
                    ygrid[:,:].reshape(1,ny,nx,1),
                    mirror_els_x_collapse.reshape(1,1,1,mirror_num_els),
                    mirror_els_y_collapse.reshape(1,1,1,mirror_num_els)))
            linelengthnan= np.isnan(linelength).any()
            avgcurvaturenan= np.isnan(avgcurvature).any()
            avgcrosscurvaturenan= np.isnan(avgcrosscurvature).any()
            thetanan=np.isnan(theta).any()

            if linelengthnan or avgcurvaturenan or avgcrosscurvaturenan or thetanan:
                sys.modules["__main__"].__dict__.update(globals())
                sys.modules["__main__"].__dict__.update(locals())
                raise ValueError("NaN! %s %s %s %s" % (linelengthnan,avgcurvaturenan,avgcrosscurvaturenan,thetanan)) # !!!

            avgcurvature_reshape=avgcurvature.reshape(1,ny,nx,mirror_num_els)
            avgcrosscurvature_reshape=avgcrosscurvature.reshape(1,ny,nx,mirror_num_els)
            linelength_reshape=linelength.reshape(1,ny,nx,mirror_num_els)

            # *** Enable integrate_over_pixel corrections 
            # (are they only necessary when x is small? ... probably.)
            #dx=abs(x[1]-x[0])
            #dy=abs(y[1]-y[0])
            integrate_over_pixel = np.ones(linelength_reshape.shape,dtype=np.bool) #np.abs(linelength_reshape) < np.min((dx,dy))/2.0
            iop_dx = integrate_over_pixel*step_size_x_collapse[np.newaxis,np.newaxis,np.newaxis,:]  # dx where iop is enabled, 0 elsewhere
            iop_dy = integrate_over_pixel*step_size_y_collapse[np.newaxis,np.newaxis,np.newaxis,:]  # dx where iop is enabled, 0 elsewhere

            # see curved_laminate_final_2d.py in heatsim2/demos
            # for heat source on surface of curved half space 
            # theory/verification
            
            if False: # !!! 
                # Python implementation... still useful to keep around for testing against GPU implementation 
                
                from curved_laminate_final_2d import nondimensionalize as curved_gf_nondim
                from curved_laminate_final_2d import evaluate as curved_gf_eval

                # This won't give quite exactly the same answer as the opencl
                # because it handles the extravolumefactor influence of the
                # crosscurvature differently, but it should be close


                nondim_params = curved_gf_nondim(source_intensity_collapse.reshape(1,1,1,mirror_num_els),rho,c,alphaz,alphaxy,avgcurvature_reshape,linelength_reshape,np.zeros((1,1,1,1),dtype='d'),t_reshape,dx=iop_dx)
                # Handle y axis (cross-curvature) ourselves
                ExtraVolumeFactorCross = 0.25*avgcrosscurvature_reshape*np.sqrt(np.pi*alphaz*t_reshape)
                ExtraVolumeFactorCross[ExtraVolumeFactorCross > 1.0] = 1.0
                ExtraVolumeFactorCross[ExtraVolumeFactorCross < -0.6]=-0.6

                Predicted_T = (1.0/(1.0+ExtraVolumeFactorCross))*curved_gf_eval(*nondim_params)
                integrate_over_pixel_bc=np.broadcast_to(integrate_over_pixel,Predicted_T.shape)
                t_reshape_bc=np.broadcast_to(t_reshape,Predicted_T.shape)
                Predicted_T[~integrate_over_pixel_bc] *= (1.0/(np.sqrt(4.0*np.pi*alphaxy*t_reshape_bc[~integrate_over_pixel_bc])))
                iop_dy_bc=np.broadcast_to(iop_dy,Predicted_T.shape) 
                Predicted_T[integrate_over_pixel_bc] *= scipy.special.erf(iop_dy_bc[integrate_over_pixel_bc]/(2.0*np.sqrt(4.0*alphaxy*t_reshape_bc[integrate_over_pixel_bc])))/iop_dy_bc[integrate_over_pixel_bc]  # divide by dy because we want integral per unit area
                
                #Predicted_T = source_intensity_collapse.reshape(1,1,1,mirror_num_els)*(2.0/(rho*c*(4*np.pi)**(3.0/2.0)*np.sqrt(alphaz*alphax*alphay)*t_reshape**(3.0/2.0)))
                ## Factor here of 0.05 is really the 0.1 determined empirically (heatsim2/test_curved_laminate_surface.py) divided by 2 because we are averaging the curvature along our line and the curvature across our line (mean curvature)
                #ExtraVolumeFactor = (0.05)*((avgcurvature_reshape+avgcrosscurvature.reshape).astype(np.float32))*np.sqrt(np.pi*alphaxy*t_reshape)
                #ExtraVolumeFactor[ExtraVolumeFactor > 1.0] = 1.0
                #ExtraVolumeFactor[ExtraVolumeFactor < -0.15]=-0.15
                #
                #Predicted_T = Predicted_T/(1.0 + ExtraVolumeFactor)
                
                #LengthFactor = 1.0+avgcurvature_reshape*np.sqrt(np.pi*alphaxy*t_reshape)/3.0
                ## Bound length factor so not less than 0.2... arbitrary choice, 
                ## but 0.0 causes action at a distance 
                #LengthFactor[LengthFactor < 0.2]=0.2
                #Predicted_T *= np.exp( -(1.0/(4.0*alphaxy*t_reshape)*(LengthFactor)*linelength_reshape**2))
            
                # Sum over source_intensity region; add into flashsourcevecs
            
                direct_calc = Predicted_T.sum(axis=3)
                flashsourcevecs.reshape(nt,ny,nx,num_sources_y*num_sources_x)[:,:,:,sourcevecidx]+=direct_calc 
                pass
            
            # This sequence of instructions is useful for debugging in IPython because
            # it lets you transfer context from pdb exception handler back to main IPython context:
            ##sys.modules["__main__"].__dict__.update(globals())
            ##sys.modules["__main__"].__dict__.update(locals())
            ##raise ValueError("FOO!")

            # GPU implementation

            this_closure = greensconvolution_greensfcn_curved(greensconvolution_params,
                                                              source_intensity_collapse.reshape(1,1,1,mirror_num_els).astype(np.float32),
                                                              linelength_reshape.astype(np.float32),
                                                              np.zeros((1,1,1,1),dtype='f'),
                                                              theta.astype(np.float32),
                                                              t_reshape.astype(np.float32),
                                                              kz,rho,c,
                                                              (3,),
                                                              avgcurvatures=avgcurvature_reshape.astype(np.float32),
                                                              avgcrosscurvatures=avgcrosscurvature_reshape.astype(np.float32),
                                                              iop_dy=iop_dy.astype(np.float32),
                                                              iop_dx=iop_dx.astype(np.float32),
                                                              ky=ky,kx=kx,
                                                              opencl_queue=opencl_queue)
            flashsourcevecs.reshape(nt,ny,nx,num_sources_y*num_sources_x)[:,:,:,sourcevecidx]=this_closure() # !!!
            this_closure=None # !!!
            #if np.isnan(flashsourcevecs.reshape(nt,ny,nx,num_sources_y*num_sources_x)[:,:,:,sourcevecidx]).any():
            #    sys.modules["__main__"].__dict__.update(globals())
            #    sys.modules["__main__"].__dict__.update(locals())
            #    raise ValueError("NaN!")
                

            if next_closure is not None:
                flashsourcevecs.reshape(nt,ny,nx,num_sources_y*num_sources_x)[:,:,:,next_assign_sourcevecidx]=next_closure()
                pass
            next_closure=this_closure
            next_assign_sourcevecidx=sourcevecidx

            pass
        pass
    
    if next_closure is not None:
        flashsourcevecs.reshape(nt,ny,nx,num_sources_y*num_sources_x)[:,:,:,next_assign_sourcevecidx]=next_closure()
        next_closure=None
        pass

    #sys.modules["__main__"].__dict__.update(globals())
    #sys.modules["__main__"].__dict__.update(locals())
    #raise ValueError("FOO!")


    opencl_queue.finish()
            
    return flashsourcevecs

def build_flash_source_vecs(ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,trange,rowscaling,num_sources_y=2,num_sources_x=2):
    # Represent source intensity variation with cos^2 pulses
    # at each corner
    # see greensfnc_doc.pdf for source magnitude scaling
    #
    # Scaled to correspond to 1 J/m^2 flash source intensity
    #
    nt=trange.shape[0]

    # rowscaling is dy*dx (formerly included dt) where dy, dx, and dt are the step sizes for the thermal image
    # column scaling is dx*dy where dx and dy are the step sizes for the source. 
    source_snr_factor=50.0 # sources are expected to have a significantly higher SNR than response data because they are relatively few points from a lot of data (blurred over many pixels) (when changing, see also build_flash_source_vecs_curved())

    # estimate the z column scaling coefficient from the 3rd usable frame 
    effective_z=np.sqrt(trange[2]*(kz/(rho*c))*np.pi)

    # Here the source is 2x2 so step size is (bnd[-1]-bnd[0])/2

    column_scaling=((y_bnd[-1]-y_bnd[0])/num_sources_y) * ((x_bnd[-1]-x_bnd[0])/num_sources_x)/(effective_z**2.0)/source_snr_factor



    scaling = rowscaling/column_scaling

    numvecs=num_sources_y*num_sources_x

    flash_source_column_scaling=column_scaling*np.ones(numvecs,dtype='d')

    (ny,nx)=xgrid.shape
    flash_source_field=build_flash_source(ygrid,xgrid,y_bnd,x_bnd,num_sources_y=num_sources_y,num_sources_x=num_sources_x)

    # ***!!!! Should perhaps implement point-source Green's function as in the 
    # curved case to simulate lateral diffusion effect of nonuniform illumination

    return (flash_source_column_scaling,(flash_source_field.reshape(numvecs,1,ny,nx)*(1.0/(np.sqrt(np.pi*rho*c*kz*trange.reshape(1,nt,1,1))))).reshape(numvecs,nt*ny*nx).transpose()*scaling)



def build_flash_source_vecs_curved(greensconvolution_params,y,x,ygrid,xgrid,y_bnd,x_bnd,stepsizemat_tile,rho,c,kz,ky,kx,trange,rowscaling,eval_linelength_avgcurvature_mirroredbox,num_sources_y,num_sources_x):
    # Represent source intensity variation with array of cos^2 pulses
    # at each corner
    # see greensfnc_doc.pdf for source magnitude scaling
    #
    # Scaled to correspond to 1 J/m^2 flash source intensity
    #

    # rowscaling is dy*dx (formerly included dt) where dy, dx, and dt are the step sizes for the thermal image
    # column scaling is dx*dy where dx and dy are the step sizes for the source. 
    # Here the source is num_sources_y x num_sources_x so step size is (bnd[-1]-bnd[0])/2

    source_snr_factor=50.0 # sources are expected to have a significantly higher SNR than response data because they are relatively few points from a lot of data (blurred over many pixels)  (when changing, see also build_flash_source_vecs())


    # estimate the z column scaling coefficient from the 3rd usable frame 
    effective_z=np.sqrt(trange[2]*(kz/(rho*c))*np.pi)

    column_scaling=((y_bnd[-1]-y_bnd[0])/num_sources_y) * ((x_bnd[-1]-x_bnd[0])/num_sources_x)/(effective_z**2.0)/source_snr_factor
    scaling = rowscaling/column_scaling

    flash_source_column_scaling=column_scaling*np.ones(num_sources_y*num_sources_x,dtype='d')

    flash_source_field=build_flash_source_curved(greensconvolution_params,y,x,ygrid,xgrid,y_bnd,x_bnd,stepsizemat_tile,rho,c,kz,ky,kx,trange,eval_linelength_avgcurvature_mirroredbox,num_sources_y,num_sources_x)
    return (flash_source_column_scaling,(flash_source_field*scaling))
    

def definereflectors(y_bnd,x_bnd,reflector_ny,reflector_nx):
    # define reflector geometries
    
    reflector_widthy=(y_bnd[-1]-y_bnd[0])/reflector_ny
    reflector_widthx=(x_bnd[-1]-x_bnd[0])/reflector_nx


    

    reflector_posy=y_bnd[0]+reflector_widthy/2.0+np.arange(reflector_ny,dtype='d')*reflector_widthy
    reflector_posx=x_bnd[0]+reflector_widthx/2.0+np.arange(reflector_nx,dtype='d')*reflector_widthx
    (reflector_ygrid,reflector_xgrid)=np.meshgrid(reflector_posy,reflector_posx,indexing='ij')
    #
    reflector_bndy=y_bnd[0]+np.arange(reflector_ny+1,dtype='d')*reflector_widthy
    reflector_bndx=x_bnd[0]+np.arange(reflector_nx+1,dtype='d')*reflector_widthx


    return (reflector_widthy,
            reflector_widthx,
            reflector_posy,
            reflector_posx,
            reflector_ygrid,
            reflector_xgrid,
            reflector_bndy,
            reflector_bndx)



# IDEA: Mirrored source vecs around tile in place of extended boundaries. 
# Basically like cosine transform. 
#
# Q: Do we need the near-surface sources or do just the deeper ones matter? 
# A: Probably just the deeper ones, but we do want to include the 
#    depth-images of shallow sources. 

def build_reflector_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,trange,
                                reflector_depth,reflector_ny,reflector_nx,gc_kernel):

    # NOTE: Should probably use try..finally or with: clause for opencl_queue to make sure it gets cleaned up properly
    
    if gc_kernel.startswith("opencl_"):
        
        greensconvolution_params.get_opencl_context()
        opencl_queue=cl.CommandQueue(greensconvolution_params.OpenCL_CTX,properties=OpenCL_GetOutOfOrderDeviceQueueProperties(greensconvolution_params.OpenCL_CTX))
        pass
    else:
        opencl_queue=None
        pass
    next_closure=None

    next_image_source_closure=None
    next_image_source_sourcevecidx=None


    nt=trange.shape[0]
    (ny,nx)=xgrid.shape
    #

    # tstar = reflector_depth**2/(np.pi*(kz/(rho*c)))
    

    #total_width_y = y_bnd[-1]-y_bnd[0]
    #total_width_x = x_bnd[-1]-x_bnd[0]
    

    # define geometries of each reflector at this depth
    (reflector_widthy,
     reflector_widthx,
     reflector_posy,
     reflector_posx,
     reflector_ygrid,
     reflector_xgrid,
     reflector_bndy,
     reflector_bndx)=definereflectors(y_bnd,x_bnd,reflector_ny,reflector_nx)

    
    
    

    
    trange_float32=trange.astype(np.float32).reshape(trange.shape[0],1,1,1)

    reflectorsourcevecs=np.zeros((ny*nx*nt,reflector_ny*reflector_nx),dtype='d',order='F')
    # Iterate over which reflector...
    for refl_yidx in range(reflector_ny):
        for refl_xidx in range(reflector_nx):
            sourcevecidx=reflector_nx*refl_yidx + refl_xidx
            #

            off_y = reflector_posy[refl_yidx]-y_bnd[0]
            off_x = reflector_posx[refl_xidx]-x_bnd[0]
            endoff_y = y_bnd[-1]-reflector_posy[refl_yidx]
            endoff_x = x_bnd[-1]-reflector_posx[refl_xidx]

            # Include image sources mirrored across borders
            # by creating a 3x3 tile array centered over our nominal
            # tile
            refl_els_y = np.array((y_bnd[0] - off_y,
                                   y_bnd[0] - off_y,
                                   y_bnd[0] - off_y,
                                   y_bnd[0] + off_y,
                                   y_bnd[0] + off_y,
                                   y_bnd[0] + off_y,
                                   y_bnd[-1] + endoff_y,
                                   y_bnd[-1] + endoff_y,
                                   y_bnd[-1] + endoff_y),dtype='d')
            refl_els_x = np.array((x_bnd[0] - off_x,
                                   x_bnd[0] + off_x,
                                   x_bnd[-1] + endoff_x,
                                   x_bnd[0] - off_x,
                                   x_bnd[0] + off_x,
                                   x_bnd[-1] + endoff_x,
                                   x_bnd[0] - off_x,
                                   x_bnd[0] + off_x,
                                   x_bnd[-1] + endoff_x),dtype='d')
            elmask=np.ones(9,dtype=np.bool)
            # Filter out sources on the opposite side
            if refl_yidx < (reflector_ny+1)/2:
                elmask[6]=False
                elmask[7]=False
                elmask[8]=False
                pass
            elif refl_yidx >= (reflector_ny+1)/2:
                elmask[0]=False
                elmask[1]=False
                elmask[2]=False
                pass
            
            if refl_xidx < (reflector_nx+1)/2:
                elmask[2]=False
                elmask[5]=False
                elmask[8]=False
                pass
            elif refl_xidx >= (reflector_nx+1)/2:
                elmask[0]=False
                elmask[3]=False
                elmask[6]=False
                pass
            
            refl_els_y=refl_els_y[elmask]
            refl_els_x=refl_els_x[elmask]
                
            refl_num_els = refl_els_x.shape[0]#9 #  number of elements above
            
            # while imageorder < np.sqrt(4.0*trange[-1]*(np.pi*(kz/(rho*c))))/reflector_depth
            # need enough images to cover entire trange
            imageorder=np.concatenate(((1,),np.arange(4,np.floor(np.sqrt(4.0*trange[-1]*(np.pi*(kz/(rho*c))))/reflector_depth),2,dtype=np.uint32)))  # first image order (1) handled by greensconvolution, rest (4, 6, 8, 10, 12, etc.) are simple method of images
            nimageorders = imageorder.shape[0]

            zvec=np.array(reflector_depth,dtype='f').reshape(1,1,1,1) 
                
            ## Iterate over all possible measurement points
            ##for meas_yidx in range(ny):

            #meas_yidx=np.arange(ny,dtype=np.uint32).reshape(1,ny,1,1)

            # Enable this print to monitor the construction process
            #print("depth=%f; refl_yidx=%d/%d; refl_xidx=%d/%d; meas_yidx=%d/%d refl_xygrid_els=%d shape=(%d,%d,%d,%d)=%d" % (reflector_depth,refl_yidx,reflector_ny,refl_xidx,reflector_nx,meas_yidx,ny,refl_xygrid_els,nx,nimageorders,trange.shape[0],refl_xygrid_els,nx*nimageorders*trange.shape[0]*refl_xygrid_els))
            # axes for computation (tidx,meas_yidx,meas_xidx,refl_num_els)
            #    for meas_xidx in range(nx):

            #meas_xidx = np.arange(nx,dtype=np.uint32).reshape(1,1,nx,1,1)
                
            #meas_xpos=x[meas_xidx]
            #meas_ypos=y[meas_yidx]
            
            # integrate at this measurement point
            # over the entire reflector
                
            # sum over first order + image sources

            
            ## rvec is scaled according to material anisotropy, per greensconvolution_integrate_anisotropic comments
            #rscaledvec_1stimage=np.sqrt(((ygrid[:,:].reshape(1,ny,nx,1)-refl_els_y.reshape(1,1,1,refl_num_els))**2*(kz/ky) + (xgrid[:,:].reshape(1,ny,nx,1)-refl_els_x.reshape(1,1,1,refl_num_els))**2*(kz/kx) + (reflector_depth*imageorder[0])**2)).astype(np.float32)
            xvec = np.sqrt((ygrid[:,:].reshape(1,ny,nx,1)-refl_els_y.reshape(1,1,1,refl_num_els))**2.0 + (xgrid[:,:].reshape(1,ny,nx,1)-refl_els_x.reshape(1,1,1,refl_num_els))**2.0).astype(np.float32)
            
            rscaledvec_no_z=np.sqrt(((ygrid[:,:].reshape(1,ny,nx,1)-refl_els_y.reshape(1,1,1,refl_num_els))**2*(kz/ky) + (xgrid[:,:].reshape(1,ny,nx,1)-refl_els_x.reshape(1,1,1,refl_num_els))**2*(kz/kx))).astype(np.float32)
                
            image_source_zposns = (reflector_depth*imageorder[1:]).astype(np.float32)
            
            # These next few lines are useful sometimes to get a parameter snapshot when debugging
            #print("testing...")
            #import hickle
            #hickle.dump((zvec,rscaledvec,trange_float32,kz,ky,kx,rho,c,dy*dx*2.0, (1,3)),"/tmp/sourcevecsparams.hkl")
            #test=greensconvolution_integrate_anisotropic(greensconvolution_params,zvec,rscaledvec,trange_float32,kz,ky,kx,rho,c,dy*dx*2.0, (1,3),kernel=gc_kernel).reshape(nx*nt)
            #print("Are all results finite: %s" % (str(np.isfinite(test).all())))
            #raise ValueError("foo")
            

            if opencl_queue is None:
                gc_vals = greensconvolution_integrate_anisotropic(greensconvolution_params,zvec[:,:,:,:],xvec[:,:,:,:],trange_float32[:,:,:,:],0.0,kz,ky,kx,rho,c,reflector_widthy*reflector_widthx*2.0, (3,),kernel=gc_kernel) 
                #  Sourcevecs now unraveled as (nt,ny,nx)  ... residual shape from gc_int_aniso is (nx,nt) so we need to transpose and reshape a bit
                #reflectorsourcevecs[(meas_yidx*nx*nt):((meas_yidx+1)*nx*nt),sourcevecidx]= 
                reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,sourcevecidx]+=gc_vals
                pass
            else: 
                this_closure = greensconvolution_integrate_anisotropic(greensconvolution_params,zvec[:,:,:,:],xvec[:,:,:,:],trange_float32[:,:,:,:],0.0,kz,ky,kx,rho,c,reflector_widthy*reflector_widthx*2.0, (3,),kernel=gc_kernel,opencl_queue=opencl_queue) 
                # Extract data from the previous iteration
                if next_closure is not None:
                    reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,next_assign_sourcevecidx]+=next_closure()
                    pass

                next_closure=this_closure
                #next_assign_meas_yidx=meas_yidx
                next_assign_sourcevecidx=sourcevecidx
                pass

            # Add in image sources  corresponding to imageorder[1:]
            #reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,sourcevecidx]+=(((reflector_widthy*reflector_widthx*2.0)*((2.0/(rho*c))/((4.0*np.pi*(kz/(rho*c))*trange_float32[:,:,:,:])**(3.0/2.0))))*(np.exp(-(rscaledvec_no_z[:,:,:,:,np.newaxis]**2.0+image_source_zposns[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]**2.0)/(4.0*(kz/(rho*c))*trange_float32[:,:,:,:,np.newaxis]))).sum(4)).sum(3)
            this_image_source_closure=None
            if nimageorders > 1:
                this_image_source_closure=greensconvolution_image_sources(greensconvolution_params,rscaledvec_no_z,trange_float32,image_source_zposns,kz,rho,c,reflector_widthy*reflector_widthx*2.0,(3,),opencl_queue=opencl_queue,kxy=np.sqrt(kx*ky))  
                pass
                
            if next_image_source_closure is not None:
                
                reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,next_image_source_sourcevecidx]+=next_image_source_closure()
                next_image_source_closure=None
                pass
            next_image_source_closure=this_image_source_closure
            next_image_source_sourcevecidx=sourcevecidx
            
            pass
            
        pass

        
    if next_closure is not None:
        # Extract data from the final iteration
        reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,next_assign_sourcevecidx]+=next_closure()
        pass

                
    if next_image_source_closure is not None:
        reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,next_image_source_sourcevecidx]+=next_image_source_closure()
        next_image_source_closure=None
        pass


    if opencl_queue is not None:
        opencl_queue.finish()
        pass

    return reflectorsourcevecs





def build_reflector_source_vecs_curved(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,stepsizemat_tile,curvmat_tile,rho,c,kz,ky,kx,trange,
                                       reflector_depth,reflector_ny,reflector_nx,gc_kernel,eval_linelength_avgcurvature_mirroredbox):

    if gc_kernel.startswith("opencl_"):
        
        greensconvolution_params.get_opencl_context()
        opencl_queue=cl.CommandQueue(greensconvolution_params.OpenCL_CTX,properties=OpenCL_GetOutOfOrderDeviceQueueProperties(greensconvolution_params.OpenCL_CTX))
        pass
    else:
        opencl_queue=None
        pass
    next_closure=None
    next_sourcevecidx=None

    next_image_source_closure=None
    next_image_source_sourcevecidx=None

    y=ygrid[:,0]
    x=xgrid[0,:]


    kxy=np.sqrt(kx*ky) # we don't really allow kx and ky to be different... 
    
    nt=trange.shape[0]
    (ny,nx)=xgrid.shape
    #

    #tstar = reflector_depth**2/(np.pi*(kz/(rho*c)))
    

    total_width_y = y_bnd[-1]-y_bnd[0]
    total_width_x = x_bnd[-1]-x_bnd[0]
    

    # define nominal geometries of each reflector at this depth
    (reflector_widthy,
     reflector_widthx,
     reflector_posy,
     reflector_posx,
     reflector_ygrid,
     reflector_xgrid,
     reflector_bndy,
     reflector_bndx)=definereflectors(y_bnd,x_bnd,reflector_ny,reflector_nx)

    
    
    if reflector_widthx < dx:
        raise ValueError("Layer reflector size in X (%f mm) is smaller than parameterization nominal step  (%f mm). Make reflector more coarse, reduce xydownsample, or use a higher resolution parameterization" % ((reflector_widthx*1000.0,dx*1000.0)))
        
    if reflector_widthy < dy:
        raise ValueError("Layer reflector size in Y (%f mm) is smaller than parameterization nominal step  (%f mm). Make reflector more coarse, reduce xydownsample, or use a higher resolution parameterization" % ((reflector_widthy*1000.0,dy*1000.0)))
                
    

    
    trange_float32=trange.astype(np.float32).reshape(trange.shape[0],1,1,1)

    reflectorsourcevecs=np.zeros((ny*nx*nt,reflector_ny*reflector_nx),dtype='d',order='F')
    # Iterate over which reflector...
    for refl_yidx in range(reflector_ny):
        for refl_xidx in range(reflector_nx):
            sourcevecidx=reflector_nx*refl_yidx + refl_xidx
            #

            

            off_y = reflector_posy[refl_yidx]-y_bnd[0]
            off_x = reflector_posx[refl_xidx]-x_bnd[0]
            endoff_y = y_bnd[-1]-reflector_posy[refl_yidx]
            endoff_x = x_bnd[-1]-reflector_posx[refl_xidx]

            # Include image sources mirrored across borders
            # by creating a 3x3 tile array centered over our nominal
            # tile
            refl_els_y = np.array((y_bnd[0] - off_y,
                                   y_bnd[0] - off_y,
                                   y_bnd[0] - off_y,
                                   y_bnd[0] + off_y,
                                   y_bnd[0] + off_y,
                                   y_bnd[0] + off_y,
                                   y_bnd[-1] + endoff_y,
                                   y_bnd[-1] + endoff_y,
                                   y_bnd[-1] + endoff_y),dtype='d')
            refl_els_x = np.array((x_bnd[0] - off_x,
                                   x_bnd[0] + off_x,
                                   x_bnd[-1] + endoff_x,
                                   x_bnd[0] - off_x,
                                   x_bnd[0] + off_x,
                                   x_bnd[-1] + endoff_x,
                                   x_bnd[0] - off_x,
                                   x_bnd[0] + off_x,
                                   x_bnd[-1] + endoff_x),dtype='d')
            elmask=np.ones(9,dtype=np.bool)
            # Filter out sources on the opposite side
            if refl_yidx < (reflector_ny+1)//2:
                elmask[6]=False
                elmask[7]=False
                elmask[8]=False
                pass
            elif refl_yidx >= (reflector_ny+1)//2:
                elmask[0]=False
                elmask[1]=False
                elmask[2]=False
                pass
            
            if refl_xidx < (reflector_nx+1)//2:
                elmask[2]=False
                elmask[5]=False
                elmask[8]=False
                pass
            elif refl_xidx >= (reflector_nx+1)//2:
                elmask[0]=False
                elmask[3]=False
                elmask[6]=False
                pass
            
            refl_els_y=refl_els_y[elmask]
            refl_els_x=refl_els_x[elmask]
                
            refl_num_els = refl_els_x.shape[0]#9 #  number of elements above
            
            # while imageorder < np.sqrt(4.0*trange[-1]*(np.pi*(kz/(rho*c))))/reflector_depth
            # need enough images to cover entire trange
            imageorder=np.concatenate(((1,),np.arange(4,np.floor(np.sqrt(4.0*trange[-1]*(np.pi*(kz/(rho*c))))/reflector_depth),2,dtype=np.uint32)))  # first image order (1) handled by greensconvolution, rest (4, 6, 8, 10, 12, etc.) are simple method of images
            #imageorder=np.concatenate(((1,),np.arange(3,np.floor(np.sqrt(4.0*trange[-1]*(np.pi*(kz/(rho*c))))/reflector_depth),2,dtype=np.uint32)))  # first image order (1) handled by greensconvolution, rest (3, 5, 7,  etc.) are Green's function images
            nimageorders = imageorder.shape[0]
            #imageorder=imageorder.reshape(1,1,1,1,nimageorders)

            zvec=np.array(reflector_depth,dtype='f').reshape(1,1,1,1) 
                
            ## Iterate over all possible measurement points
            ##for meas_yidx in range(ny):
            meas_yidx=np.arange(ny,dtype=np.uint32).reshape(1,ny,1,1)
            # Enable this print to monitor the construction process
            #print("depth=%f; refl_yidx=%d/%d; refl_xidx=%d/%d; meas_yidx=%d/%d refl_xygrid_els=%d shape=(%d,%d,%d,%d)=%d" % (reflector_depth,refl_yidx,reflector_ny,refl_xidx,reflector_nx,meas_yidx,ny,refl_xygrid_els,nx,nimageorders,trange.shape[0],refl_xygrid_els,nx*nimageorders*trange.shape[0]*refl_xygrid_els))
            # axes for computation (tidx,meas_yidx,meas_xidx,refl_num_els)
            #    for meas_xidx in range(nx):
            meas_xidx = np.arange(nx,dtype=np.uint32).reshape(1,1,nx,1,1)
                
            #meas_xpos=x[meas_xidx]
            #meas_ypos=y[meas_yidx]
            
            # integrate at this measurement point
            # over the entire reflector
                
            # sum over first order + image sources

            # getting 44x2x787x1155 executing in .75 seconds
            
            # rvec is scaled according to material anisotropy, per greensconvolution_integrate_anisotropic comments

            # ****!!!! REALLY SHOULD HANDLE PERPENDICULAR CURVATURES HERE TOO !!!*** (they're calculated as avgcrosscurvature, below, but not used.)
            # ****!!!  REALLY SHOULD CONSIDER AMPLITUDE EFFECT OF CURVATURE ON 1D CONDUCTION FROM THE TOP SURFACE DOWN TO THE REFLECTION PLANE,
            #          NOT JUST THE 3D CONDUTION BACK TO THE SURFACE

            (linelength_noz_unscaled,avgcurvature,avgcrosscurvature,theta) = (
                eval_linelength_avgcurvature_mirroredbox(
                    # box corners:
                    x_bnd[0],y_bnd[0],
                    x_bnd[-1],y_bnd[-1],
                    xgrid[:,:].reshape(1,ny,nx,1),
                    ygrid[:,:].reshape(1,ny,nx,1),
                    refl_els_x.reshape(1,1,1,refl_num_els),
                    refl_els_y.reshape(1,1,1,refl_num_els)))

            linelengthnan= np.isnan(linelength_noz_unscaled).any()
            avgcurvaturenan= np.isnan(avgcurvature).any()
            avgcrosscurvaturenan= np.isnan(avgcrosscurvature).any()
            thetanan=np.isnan(theta).any()

            if linelengthnan or avgcurvaturenan or avgcrosscurvaturenan or thetanan:
                sys.modules["__main__"].__dict__.update(globals())
                sys.modules["__main__"].__dict__.update(locals())
                raise ValueError("NaN! %s %s %s %s" % (linelengthnan,avgcurvaturenan,avgcrosscurvaturenan,thetanan)) # !!!
            

            avgcurvature_float=avgcurvature.astype(np.float32)

            #linelength_scaled_float = np.sqrt((linelength_noz_unscaled**2)*(kz/kx) + (reflector_depth*imageorder[0])**2).astype(np.float32)

            linelength_noz_unscaled_float=linelength_noz_unscaled.astype(np.float32)
            
            linelength_scaled_noz_float = (linelength_noz_unscaled*np.sqrt(kz/kx)).astype(np.float32)
                            
            image_source_zposns = (reflector_depth*imageorder[1:]).astype(np.float32)
            #print("testing...")
            #import hickle
            #hickle.dump((zvec,rscaledvec,trange_float32,kz,ky,kx,rho,c,dy*dx*2.0, (1,3)),"/tmp/sourcevecsparams.hkl")
            #test=greensconvolution_integrate_anisotropic(greensconvolution_params,zvec,rscaledvec,trange_float32,kz,ky,kx,rho,c,dy*dx*2.0, (1,3),kernel=gc_kernel).reshape(nx*nt)
            #print("Are all results finite: %s" % (str(np.isfinite(test).all())))
            #raise ValueError("foo")
            

            # source calculation
            # Scale reflector_widthy*reflector_widthx according to local x & y step size scaling and depth (via curvature) 
            # reflector_area=reflector_widthy*reflector_widthx
            reflector_pixels_y = (y >= (reflector_posy[refl_yidx]-reflector_widthy/2.0)) & (y <= (reflector_posy[refl_yidx]+reflector_widthy/2.0))
            reflector_pixels_x = (x >= (reflector_posx[refl_xidx]-reflector_widthx/2.0)) & (x <= (reflector_posx[refl_xidx]+reflector_widthx/2.0))
            reflector_pixels_widthy = stepsizemat_tile[reflector_pixels_y,:,:][:,reflector_pixels_x,1]
            reflector_pixels_widthx = stepsizemat_tile[reflector_pixels_y,:,:][:,reflector_pixels_x,0]
            reflector_pixels_area = np.sum(reflector_pixels_widthy*reflector_pixels_widthx) # sum (dx*dy) over both axes to get total area 
            # ... but that is the area of the pixels, which don't actually
            # line up perfectly with the reflector. Need to correct:
            reflector_pixels_nominalarea = dy*np.count_nonzero(reflector_pixels_y) * dx*np.count_nonzero(reflector_pixels_x)

            surface_reflector_area = reflector_pixels_area * (reflector_widthy*reflector_widthx)/reflector_pixels_nominalarea

            # Correction of reflector area according to depth below surface.
            # The above calculated reflector_area is the area on the surface, 
            # but the surface is curved such that (positive curvature) 
            # deeper reflectors have more area or (negative curvature) 
            # deeper reflectors have less area. 
            # 
            # Controlling parameter here is the sum of the principal curvatures
            # i.e. the trace of the curvature matrix
            # averaged over the reflector area
            
            sumcurvs = np.trace(curvmat_tile[reflector_pixels_y,:,:,:][:,reflector_pixels_x,:,:],axis1=2,axis2=3)
            avgsumcurvs = np.mean(sumcurvs)

            if np.isnan(avgsumcurvs):
                avgsumcurvs=0.0
                sys.stderr.write("build_reflector_source_vecs_curved WARNING at refl_xidx=%d, refl_yidx=%d: No pixels under reflector area in curvature calculation. Assuming zero curvature.\n" % (refl_xidx,refl_yidx))
                pass
                
            extra_area_factor = avgsumcurvs*reflector_depth
            # extra_area_factor determined geometrically

            # We don't want this curvature correction factor to get too 
            # close to -1.0
            # similar to greensfcn_curved.c, 
            # bound it at -0.7
            if extra_area_factor < -0.7:
                extra_area_factor=-0.7
                pass

            # Likewise upper bound it at 2.0 so the biggest scaling
            # can be factor of 3 (when the reflector is twice as deep
            # (or more) as the radius of curvature on the surface)
            if extra_area_factor > 2.0:
                extra_area_factor=2.0
                pass
            
            reflector_area = (1.0+extra_area_factor)*surface_reflector_area
            #reflector_area = reflector_widthy*reflector_widthx
            if opencl_queue is None:
                gc_vals = greensconvolution_integrate_anisotropic(greensconvolution_params,zvec[:,:,:,:],linelength_noz_unscaled_float[:,:,:,:],trange_float32[:,:,:,:],0.0,kz,ky,kx,rho,c,reflector_area*2.0, (3,),avgcurvatures=avgcurvature_float,kernel=gc_kernel)
                #  Sourcevecs now unraveled as (nt,ny,nx)  ... residual shape from gc_int_aniso is (nx,nt) so we need to transpose and reshape a bit
                #reflectorsourcevecs[(meas_yidx*nx*nt):((meas_yidx+1)*nx*nt),sourcevecidx]= 
                reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,sourcevecidx]+=gc_vals
                pass
            else: 
                this_closure = greensconvolution_integrate_anisotropic(greensconvolution_params,zvec[:,:,:,:],linelength_noz_unscaled_float,trange_float32[:,:,:,:],0.0,kz,ky,kx,rho,c,reflector_area*2.0, (3,),avgcurvatures=avgcurvature_float,kernel=gc_kernel,opencl_queue=opencl_queue)

                #this_res=this_closure()
                #reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,sourcevecidx]+=this_res
                #
                #this_closure=None
                #
                #if np.isnan(reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,sourcevecidx]).any():
                #    sys.modules["__main__"].__dict__.update(globals())
                #    sys.modules["__main__"].__dict__.update(locals())
                #    raise ValueError("NaN!") # !!!

                # Extract data from the previous iteration
                if next_closure is not None:
                    reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,next_sourcevecidx]+=next_closure()
                    pass
                    
                next_closure=this_closure
                #next_assign_meas_yidx=meas_yidx
                next_sourcevecidx=sourcevecidx
                pass

            #sys.modules["__main__"].__dict__.update(globals())
            #sys.modules["__main__"].__dict__.update(locals())
            #raise ValueError("FOO!")
            ##import pdb
            ##pdb.set_trace()


            
            # Add in image sources  corresponding to imageorder[1:]
            #reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,sourcevecidx]+=(((reflector_widthy*reflector_widthx*2.0)*((2.0/(rho*c))/((4.0*np.pi*(kz/(rho*c))*trange_float32[:,:,:,:])**(3.0/2.0))))*(np.exp(-(rscaledvec_no_z[:,:,:,:,np.newaxis]**2.0+image_source_zposns[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]**2.0)/(4.0*(kz/(rho*c))*trange_float32[:,:,:,:,np.newaxis]))).sum(4)).sum(3)
            
            this_image_source_closure=None
            if nimageorders > 1:
                this_image_source_closure=greensconvolution_image_sources(greensconvolution_params,linelength_scaled_noz_float,trange_float32,image_source_zposns,kz,rho,c,reflector_area*2.0,(3,),avgcurvatures=avgcurvature_float,kxy=kxy,opencl_queue=opencl_queue)  
                pass
                
            if next_image_source_closure is not None:
                
                reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,next_image_source_sourcevecidx]+=next_image_source_closure()
                next_image_source_closure=None
                pass
            next_image_source_closure=this_image_source_closure
            next_image_source_sourcevecidx=sourcevecidx
            

            pass
            
        pass

    #if reflector_depth < 0.4e-3:
    #    import pdb
    #    pdb.set_trace()
    #    pass
        
    if next_closure is not None:
        # Extract data from the final iteration
        reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,next_sourcevecidx]+=next_closure()
        pass


    if next_image_source_closure is not None:
        reflectorsourcevecs.reshape(nt,ny,nx,reflector_ny*reflector_nx)[:,:,:,next_image_source_sourcevecidx]+=next_image_source_closure()
        next_image_source_closure=None
        pass


    if opencl_queue is not None:
        opencl_queue.finish()
        pass

    return reflectorsourcevecs







def build_all_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,dt,trange,reflectors,return_conditions="None",gc_kernel="opencl_interpolator",eval_linelength_avgcurvature_mirroredbox=None,curvmat_tile=None,stepsizemat_tile=None,num_sources_y=2,num_sources_x=2):
    # *** NOTE: dt parameter just added ***

    # reflectors is a tuple of (z,ny,nx) tuples representing
    # possible z values for reflectors and how many y and x pieces
    # they should be split into.
    # it should be ordered from the back surface towards the
    # front surface. 
    
    y=ygrid[:,0]
    x=xgrid[0,:]
    
    rowscaling=dy*dx
    
    print("Flash source: %d by %d" % (num_sources_x,num_sources_y))
    if eval_linelength_avgcurvature_mirroredbox is None:
        # flat case
        #assert(num_sources_y==2 and num_sources_x==2)  # flat case not implemented for anything but two sources in each axis (four sources total)
        (flashsourcecolumnscaling,flashsourcevecs)=build_flash_source_vecs(ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,trange,rowscaling,num_sources_y=num_sources_y,num_sources_x=num_sources_x)
        pass
    else:
        (flashsourcecolumnscaling,flashsourcevecs)=build_flash_source_vecs_curved(greensconvolution_params,y,x,ygrid,xgrid,y_bnd,x_bnd,stepsizemat_tile,rho,c,kz,ky,kx,trange,rowscaling,eval_linelength_avgcurvature_mirroredbox,num_sources_y,num_sources_x)        
        pass

    reflectorsourcevecs=[]
    reflectorcolumnscaling=[]
    tstars=[]
    conditions=[]   # condition number for these vectors alone
    prevconditions=[]  # condition number for these vectors + previous vectors
    prevscaledconditions=[]  # condition number for these vectors + previous vectors scaled
    depths=[]

    prevsourcevecs=None
    for (reflector_depth,reflector_ny,reflector_nx) in reflectors:
        print("Reflector: Depth=%f" % (reflector_depth))
        #import pdb
        #pdb.set_trace()
        #sourceveccolumnscaling = ((y_bnd[-1]-y_bnd[0])/reflector_ny) * ((x_bnd[-1]-x_bnd[0])/reflector_nx)/(reflector_depth**3.0)

        # The original (unscaled) forward matrix has units of deg K/ (J/m^2)
        # after scaling by rowscaling=dx*dy and dividing by 
        # columnscaling = src_dx*src_dy/z^2
        # it has units of deg K * m^2 / (J/m^2) 
        # 
        # The Tikhonov parameter should have these same units
        # A larger Tikhonov parameter gives less noise (in J/m^2)
        # per unit camera NETD. 
        # So in that vein, let's look at the reciprocal of the Tikhonov
        # parameter... let's call it 1/T
        # 1/T   has units of J/m^2 / (deg K * m^2 of depth) 
        # 
        # It represents energy uncertainty over temperature uncertainty
        # multiply this by a temperature change at a particular depth
        # and it gives you a source intensity (J/m^2)
        # 
        # ... This is all very well but it isn't going to be 
        # frame-rate independent... A higher frame rate gives more 
        # averaging or less noise so the Tikhonov parameter can be lower
        # (i.e. 1/T can be higher). 
        #
        # For a measurement over a given time period t, 
        # the number of frames n=t*framerate
        # Temperature uncertainty deltaT = NETD/sqrt(t*framerate)
        # for a measurement at a characteristic depth z, 
        # the time involved in the measurement is roughly t=z^2/(pi*alpha)
        # so temperature uncertainty is deltaT=(NETD/sqrt(framerate))/(z/sqrt(pi*alpha))
        #  or deltaT = NETD * (sqrt(pi*alpha)/sqrt(framerate)) / z
        # which has units of K
        # 
        # i.e. temperature uncertainty effect from camera averaging 
        # scales with NETD and  (sqrt(pi*alpha)/sqrt(framerate))
        # and inversely with z

        # Propose: 
        # Don't just represent temperature uncertainty as NETD
        # Instead represent it as NETD * (sqrt(pi*alpha)/sqrt(framerate)) / z
        # 
        # Tikhonov parameter should be smaller at a higher framerate;
        # 1/T should be higher 
        # So instead of dividing columns (source intensities) by 
        # src_dx*src_dy/z^2, 
        # let's divide them by src_dx*src_dy*(sqrt(pi*alpha)/sqrt(framerate))/z^3
        # Now with higher framerate we will have bigger singular values 
        # and we will therefore chop off less. 
        # (REJECTED; rationale is that it's OK to need to tweak the 
        # Tikhonov parameter after a frame rate adjustment as we don't
        # know if the frame rate adjustment goal was to increase 
        # sensitivity or decrease noise)
        

        sourceveccolumnscaling = ((y_bnd[-1]-y_bnd[0])/reflector_ny) * ((x_bnd[-1]-x_bnd[0])/reflector_nx)/(reflector_depth**2.0)
        sourcevecscalefactor=rowscaling/sourceveccolumnscaling

        if eval_linelength_avgcurvature_mirroredbox is None:
            # flat case
            thesesourcevecs=build_reflector_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,trange,reflector_depth,reflector_ny,reflector_nx,gc_kernel)*sourcevecscalefactor
            pass
        else:
            thesesourcevecs=build_reflector_source_vecs_curved(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,stepsizemat_tile,curvmat_tile,rho,c,kz,ky,kx,trange,reflector_depth,reflector_ny,reflector_nx,gc_kernel,eval_linelength_avgcurvature_mirroredbox)*sourcevecscalefactor

            pass
        sourceveccolumnscaling_array=sourceveccolumnscaling*np.ones(reflector_ny*reflector_nx,dtype='d')

        ###***!!! This SVD call may be somewhat time consuming... is it necessary?
        if return_conditions=="incremental" or return_conditions=="full":
            (u,s,v)=np.linalg.svd(thesesourcevecs,full_matrices=False)
            
            conditions.append(max(s)/min(s))
            pass
        
        depths.append(reflector_depth)
        tstars.append(reflector_depth**2/(np.pi*(kz/(rho*c))))

        if return_conditions=="full":
            if prevsourcevecs is not None:
                concatenated=np.concatenate((prevsourcevecs,thesesourcevecs),axis=1)
                ###***!!! This SVD call may be quite time consuming... is it necessary? 
                (uprev,sprev,vprev)=np.linalg.svd(concatenated,full_matrices=False)
                prevconditions.append(max(sprev)/min(sprev))

                prevscaledconditions.append(scaledcondition(concatenated))
                
                pass
            else:
                prevconditions.append(None)
                prevscaledconditions.append(None)
                pass
            pass
        
        reflectorsourcevecs.append(thesesourcevecs)
        reflectorcolumnscaling.append(sourceveccolumnscaling_array)


        prevsourcevecs=thesesourcevecs
        pass


    if return_conditions=="incremental":
        
        return (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,None,None)
    elif return_conditions=="full":
        return (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions)
    else:
        return (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,None,None,None)

    pass


def NaN_in_sourcevecs(reflectorsourcevecs):
    # sourcevecs can be huge (many gigabytes)
    # .. don't want to make a bool copy just 
    # to find NaNs. 
    for layernum in range(len(reflectorsourcevecs)):
        for sourcevecidx in range(reflectorsourcevecs[layernum].shape[1]):
            if np.isnan(reflectorsourcevecs[layernum][:,sourcevecidx]).any():
                return True
            pass
        pass
    return False

    

        
