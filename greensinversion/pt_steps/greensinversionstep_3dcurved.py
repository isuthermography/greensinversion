import sys
import os
import re
import copy
import posixpath
# import json
import ast

import dg_units
import dg_eval
import dg_file as dgf
import dataguzzler as dg
import dg_metadata as dgm

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

from limatix import dc_value

import numpy as np
from matplotlib import pyplot as pl

from spatialnde.coordframes import coordframe
from spatialnde.dataguzzler.dg_3d import ndepart_from_dataguzzler_wfm


#import heatsim2

# !!! Should add back layer (infinite or perhaps quartered) sheet at assumed thickness) which 
# we can solve for the intensity of. 
#  * Or perhaps we allow an infinite sheet at every layer?
#  * Either would help to remove the lousy results at the sample back wall
#  * Better yet have each border element be integrated over twice its
#    area off-screen (IMPLEMENTED THIS)
#


from greensconvolution.greensconvolution_calc import read_greensconvolution


import greensinversion
import greensinversion.fillholes
from greensinversion import build_gi_grid
from greensinversion import define_flat_inversion
from greensinversion import define_curved_inversion
from greensinversion import NotANumberError
#import greensinversion.tile_rectangle
#from greensinversion.heatsim_calc import heatsim_calc


#if parse_version(np.version.version) < parse_version("1.12"):
def eigvals_broadcast_nans(A):
    numdims=len(A.shape)
    if numdims==2:
        return np.linalg.eigvals(A)
        
    unwrapped_dims = (np.prod(A.shape[:-2]),A.shape[-2],A.shape[-1])
    A_unwrapped=A.reshape(*unwrapped_dims)
    
    ret=np.empty((unwrapped_dims[0],unwrapped_dims[1]),dtype='d')
    for cnt in range(unwrapped_dims[0]):
        if np.isnan(A_unwrapped[cnt,:,:]).any():
            ret[cnt,:]=np.NaN
            pass
        else:
            ret[cnt,:]=np.linalg.eigvals(A_unwrapped[cnt,:,:])
            pass
        pass
    return ret.reshape(*A.shape[:-1])

def fast2x2evals(A):

    # assume broadcasting over 2 previous axes
    evals=np.empty((A.shape[0],A.shape[1],2),dtype='d')

    # evals = (trace +/- sqrt(trace^2-4det))/2
    trace=A[:,:,0,0]+A[:,:,1,1]
    det = A[:,:,0,0]*A[:,:,1,1] - A[:,:,0,1]*A[:,:,1,0]
    evals[:,:,0] = (trace + np.sqrt(trace**2.0 - 4.0*det))/2.0
    evals[:,:,1] = (trace - np.sqrt(trace**2.0 - 4.0*det))/2.0
    return evals





# New material properties:
# Heat capacity and density (i.e. volumetric heat capacity)
# based on THERMAL PROPERTIES OF CARBON FIBER-EPOXY
# COMPOSITES WITH DIFFERENT FABRIC WEAVES by R. Joven,
# R. Das, A. Ahmed, P. Roozbehjavan, B. Minaie, figure 6
# we get a typical room temperature Cp of 850 J/(kg*deg K)
# Specifically for T300 3K/977-2PW we get 850 J/(kg * deg K)
# and from SAE composites materials handbook, volume 2,
# section 2.3.1.14, properties of T300 3K/9772 plain weave
# nominal composite density of 1.54-1.57 g/cm^3.
# Splitting the difference, use 1.555 g/cm^3
#
# Citation: Composite Materials Handbook-17. (2012). Composite Materials Handbook, Volume 2 - Polymer Matrix Composites - Materials Properties. SAE International on behalf of CMH-17, a division of Wichita State University. Online version available at: http://app.knovel.com/hotlink/toc/id:kpCMHVPM0J/composite-materials-handbook/composite-materials-handbook

dg_units.units_config("insert_basic_units");


# Because we rununlocked we can't actually access or modify the XML document directly
def rununlocked(_dest_href,dc_dgsfile_href,dc_density_numericunits,dc_specificheat_numericunits,dc_alphaz_numericunits,dc_alphaxy_numericunits,dc_nominal_lamina_thickness_numericunits,dc_lamina_thickness_numericunits,dc_numlayers_numericunits,dc_inversion_tile_size_y_numericunits,dc_inversion_tile_size_x_numericunits,dc_inversion_channel_str,dc_inversion_startframe_numericunits,dc_flashtime_numericunits,dc_inversion_reflectors_str,xydownsample_numericunits,tikparam_numericunits,dc_cadmodel_channel_str,dc_scalefactor_x_numericunits=dc_value.numericunitsvalue(1.0,"Unitless"),dc_scalefactor_y_numericunits=dc_value.numericunitsvalue(1.0,"Unitless"),dc_numplotrows_int=3,dc_numplotcols_int=4,do_singlestep_bool=True,dc_holesadjusted_xmltree=None,dc_source_approx_dx_numericunits=None,dc_source_approx_dy_numericunits=None):

    tikparam=tikparam_numericunits.value()

    dc_prefix_str="greensinversion_"



    reslist=[]

   
    if tikparam==0.0:
        tikparam=None  # 0 and disabled are equivalent
        pass
        
 
    #rho=float(1.555e3) # kg/m^3
    #c=float(850.0) # J/(kg* deg K)

    rho=dc_density_numericunits.value('kg/m^3')
    c=dc_specificheat_numericunits.value('J/(kg*K)')

    # alpha units are m^2/s
    #alphaz=float(.54e-6) # average value from measurements (Thermal_Properties.ods 11/25/15, averaging in-plane value from 90deg specimen and flash method values)
    alphaz=dc_alphaz_numericunits.value('m^2/s')

    #alphaxy=float(3.00e-6) # best evaluation based on Thermal_Properties.ods 3/19/16 based on 0/90 and quasi-isotropic layups
    alphaxy=dc_alphaxy_numericunits.value('m^2/s')


    # Lamina thickness based on thermal_properties.ods average thickness of 8.05 mm for 3(?) layers of 16 plies
    # nominal_lamina_thickness=8.05e-3/(3.0*16.0)
    nominal_lamina_thickness=dc_nominal_lamina_thickness_numericunits.value('m')

    # Load input file
    # NOTE: When changing input file: 
    #  1. Verify flashtime. Adjust as appropriate
    #  2. Verify startframe. Adjust as appropriate
    #  3. Execute file load code (below) and evaluate
    #    a) XStepMeters (must match dx)
    #    b) YStepMeters (must match dy)
    #    c) TStep (must match dt)
    #    d) bases[2][startframe]-flashtrigtime  (must match t0)
    #    e) bases[2][startframe:endframe].shape[0] (must match nt)
    #  4. Adjust dx, dy, dt, t0, and/or nt to satisfy above criteria
    #  5. Once adjusted, assert()s below should pass. 
    
    inputfile=dc_dgsfile_href.getpath()  # was "/tmp/CA-1_Bottom_2015_11_19_undistorted_orthographic.dgs"
    (inputfile_basename,inputfile_ext) = posixpath.splitext(dc_dgsfile_href.get_bare_unquoted_filename())

    if inputfile_ext==".bz2" or inputfile_ext==".gz":  # .dgs.bz2 or .dgs.gz
        orig_inputfile_basename = inputfile_basename
        inputfile_basename=posixpath.splitext(orig_inputfile_basename)[0]
        inputfile_ext = posixpath.splitext(orig_inputfile_basename)[1] + inputfile_ext
        pass
    #flashtrigtime=0.2 # seconds -- from pequod system
    #flashtime=flashtrigtime+1.0/100.0 # add 1/100th second delay of flash peak (wild guess!)
    flashtime=dc_flashtime_numericunits.value('s')

    #channel="DiffStack"
    channel=dc_inversion_channel_str
    # frame #165: Time relative to trigger = bases[2][165]-flashtrigtime
    #                                      = 0.052869999999999973
    #startframe=13  # zero-based, not one-based
    startframe=int(round(dc_inversion_startframe_numericunits.value('unitless')))

    (junkmd,wfmdict)=dgf.loadsnapshot(inputfile,memmapok=True)

    channel3d = "Proj" + dc_inversion_channel_str[:-4] # Proj + diffstack channel with _tex stripped
    objframe=coordframe()
    (obj, TexChanPrefix) = ndepart_from_dataguzzler_wfm(wfmdict[channel3d],wfmdict,objframe)


    channel_weights=channel+"_weights"
    if channel_weights not in wfmdict:
        channel_weights = None
        pass


    (ndim,DimLen,IniVal,Step,bases)=dg_eval.geom(wfmdict[channel],raw=True)
    (ndim,Coord,Units,AmplCoord,AmplUnits)=dg_eval.axes(wfmdict[channel],raw=True)
    XIniValMeters=dc_value.numericunitsvalue(IniVal[0],Units[0]).value('m')
    YIniValMeters=dc_value.numericunitsvalue(IniVal[1],Units[1]).value('m')

    # Apply scaling factor to XStepMeters (note that Coord, above, is not corrected!!!)
    XStepMeters=dc_value.numericunitsvalue(Step[0],Units[0]).value('m')*dc_scalefactor_x_numericunits.value()
    YStepMeters=dc_value.numericunitsvalue(Step[1],Units[1]).value('m')*dc_scalefactor_y_numericunits.value()
    TStep=Step[2]

    (saturation_fraction,saturation_map)=greensinversion.saturationcheck(wfmdict[channel].data.transpose((2,1,0)),startframe) 
    if saturation_fraction > .2: 
        raise ValueError("greensinversionstep: ERROR: %.1f%% of pixels are saturated at least once beyond start frame!" % (saturation_fraction*100.0))
    if saturation_fraction > .02:
        sys.stderr.write("greensinversionstep: WARNING: %.1f%% of pixels are saturated at least once beyond start frame!\n" % (saturation_fraction*100.0))
        pass

    # Apply spatial downsampling to keep inversion complexity under control
    #xydownsample=2

    xydownsample=int(round(xydownsample_numericunits.value("unitless")))

    # reflectors is a tuple of (z,ny,nx) tuples representing
    # possible z values for reflectors and how many y and x pieces
    # they should be split into.
    # it should be ordered from the back surface towards the
    # front surface. 

    # reflectors is (depth, reflector_ny,reflector_nx)

    # # need pre-calculation of z_bnd to determine reflectors
    # z_bnd=np.arange(nz+1,dtype='d')*dz  # z boundary starts at zero

    # reflectors=( (z_bnd[15],4,4),
    #              (z_bnd[9],4,4),
    #              (z_bnd[5],6,6),
    #              (z_bnd[2],10,10))
    
    reflectors_float=ast.literal_eval(dc_inversion_reflectors_str)
    
    # reflectors can just be reflectors_float but this is here to avoid
    # some temporary recalculations 3/29/16
    reflectors=tuple([ (np.float64(reflector[0]),reflector[1],reflector[2]) for reflector in reflectors_float])
    deepest_tstar = reflectors[0][0]**2.0/(np.pi*alphaz)

    endframe = np.argmin(np.abs(bases[2]-flashtime-deepest_tstar*2.0))   # see also generateinversionsteps() call to timelimitmatrix()

    # step sizes for inversion
    dx=XStepMeters*1.0*xydownsample
    dy=YStepMeters*1.0*xydownsample
    dt=TStep
    t0=bases[2][startframe]-flashtime
    nt=bases[2][startframe:endframe].shape[0]

    dz=nominal_lamina_thickness  # use nominal value so we don't recalculate everything for each sample
    
    # These now satisfied by definition
    #assert(XStepMeters==dx)
    #assert(YStepMeters==dy)
    #assert(TStep==dt)
    #assert(bases[2][startframe]-flashtrigtime==t0)  # Start time matches  NOTE.... CHANGED FROM flashtrigtime to flashtime
    #assert(bases[2][startframe:].shape[0]==nt) # Number of frames match

    # These are parameters for the reconstruction, not the expermental data
        
    #nz=16   # NOTE: nz*dz should match specimen thickness
    nz=int(round(dc_numlayers_numericunits.value('unitless')))
    
    # size of each tile for tiled inversion
    #maxy=38.0e-3
    #maxx=36.0e-3
    maxy=dc_inversion_tile_size_y_numericunits.value('m')
    maxx=dc_inversion_tile_size_x_numericunits.value('m')

    source_approx_dy=None
    source_approx_dx=None
    
    if dc_source_approx_dy_numericunits is not None:
        source_approx_dy=dc_source_approx_dy_numericunits.value('m')
        pass

    if dc_source_approx_dx_numericunits is not None:
        source_approx_dx=dc_source_approx_dx_numericunits.value('m')
        pass

    greensconvolution_params=read_greensconvolution()

    greensconvolution_params.get_opencl_context("GPU",None)
    
    

    #(kx,ky,kz,
    # ny,nx,
    # z,y,x,
    # zgrid,ygrid,xgrid,
    # z_bnd,y_bnd,x_bnd,
    # flashsourcevecs,
    # reflectorsourcevecs,
    # depths,tstars,
    # conditions,prevconditions,prevscaledconditions,
    # rowselects,
    # inversions,
    # inversionsfull,
    # inverses,
    # nresults,
    # ss_rowselects,
    # ss_inversions,
    # ss_inversionsfull,
    # ss_inverses,
    # ss_nresults)=greensinversion.greensinversion_lookup(cache_dir,rho,c,alphaz,alphaxy,dz,dy,dx,nz,maxy,maxx,t0,dt,nt,reflectors)

    kx=alphaxy*rho*c
    ky=alphaxy*rho*c
    kz=alphaz*rho*c

    trange=t0+np.arange(nt,dtype='d')*dt


    gi_params=(rho,c,alphaz,alphaxy,dy,dx,maxy,maxx,t0,dt,nt,reflectors,
     trange,greensconvolution_params)
    
    flat_gi_grid = build_gi_grid(dy,maxy,
                                 dx,maxx)
    (ny,nx,y,x,ygrid,xgrid,y_bnd,x_bnd) = flat_gi_grid

    
    num_sources_y=2
    num_sources_x=2

    if source_approx_dy is not None or source_approx_dx is not None:
        (num_sources_y,num_sources_x) = greensinversion.num_sources(y,x,y_bnd,x_bnd,source_approx_dy,source_approx_dx)
        pass





    # can view individual source maps with
    # reflectorsourcevecs[:,0].reshape(ny,nx,nt),
    # e.g. imshow(reflectorsourcevecs[:,5].reshape(ny,nx,nt)[:,:,200])


    #pl.figure(1)
    #pl.clf()
    #pl.imshow(reflectorsourcevecs[0][:,5].reshape(ny,nx,nt)[:,:,200])


    
    #pl.figure(2)
    #pl.clf()
    #pl.imshow(reflectorsourcevecs[1][:,5].reshape(ny,nx,nt)[:,:,200])
    


    # To plot: 
    # loglog(trange+dt/2,T[20,20,:])
    # imshow(T[:,:,200]

    # Break object into tiles, perform inversion on each tile

    (minyminx_corners,yranges,xranges,contributionprofiles)=greensinversion.build_tiled_rectangle(ny,nx,dy,dx,reflectors,wfmdict[channel].data.transpose((2,1,0)),xydownsample)



    inputmats = [ wfmdict[channel].data[(xidx*xydownsample):((xidx+nx)*xydownsample):xydownsample,(yidx*xydownsample):((yidx+ny)*xydownsample):xydownsample,startframe:endframe].transpose((2,1,0)) for (yidx,xidx) in minyminx_corners ]  # transpose to convert dataguzzler axis ordering (x,y,t) to greensinversion ordering (t,y,x)

    print("Filling holes...")
    inputmats_holesfilled = [ greensinversion.fillholes.fillholes_flat(inputmat) for inputmat in inputmats ]
    print("Done filling holes.")

    parallelevaluate=False   # GPU is currently slightly SLOWER here (WHY?) so we don't use it
    if parallelevaluate:
        inversionevalfunc=greensinversion.inversion.parallelperforminversionsteps
        OpenCL_CTX=greensconvolution_params.get_opencl_context()   #greensinversion.inversion.Get_OpenCL_Context()
        pass
    else:
        inversionevalfunc=greensinversion.inversion.serialperforminversionsteps
        OpenCL_CTX=None
        pass


    print("Evaluating curvatures")
    hires_factor=2
    curvmat=obj.implpart.surfaces[0].intrinsicparameterization.interpolate_curvature(obj.implpart.surfaces[0],wfmdict[channel].data.shape[1]/xydownsample,wfmdict[channel].data.shape[0]/xydownsample)

    # curvmat is uv_channame ny x nx x 2x2 matrix representing the shape operator

    curvmat_hires=obj.implpart.surfaces[0].intrinsicparameterization.interpolate_curvature(obj.implpart.surfaces[0],wfmdict[channel].data.shape[1]*hires_factor//xydownsample,wfmdict[channel].data.shape[0]*hires_factor//xydownsample)

    # Set unknown curvatures to zero
    curvmat[np.isnan(curvmat)]=0.0
    curvmat_hires[np.isnan(curvmat_hires)]=0.0


    # These are only nominal physical sizes (in terms of nominal dx and dy of parameterization)
    curvmat_sizex = dx * wfmdict[channel].data.shape[0]/xydownsample
    curvmat_sizey = dy * wfmdict[channel].data.shape[1]/xydownsample

    print("Determining maximum principal curvatures")
    #maxabs_princcurvs = np.max(np.abs(eigvals_broadcast_nans(curvmat)),2)
    maxabs_princcurvs = np.max(np.abs(fast2x2evals(curvmat)),2)

    print("Evaluating step sizes")
    stepsizemat=obj.implpart.surfaces[0].intrinsicparameterization.interpolate_stepsizes(obj.implpart.surfaces[0],wfmdict[channel].data.shape[1]//xydownsample,wfmdict[channel].data.shape[0]//xydownsample)
    
    stepsizemat_hires=obj.implpart.surfaces[0].intrinsicparameterization.interpolate_stepsizes(obj.implpart.surfaces[0],wfmdict[channel].data.shape[1]*hires_factor//xydownsample,wfmdict[channel].data.shape[0]*hires_factor//xydownsample)

    # Fill in invalid stepsizes sith dx,dy
    ssm_xy_nelem=stepsizemat.shape[0]*stepsizemat.shape[1]
    ssm_nan_dx=np.isnan(stepsizemat.reshape(ssm_xy_nelem,2)[:,0])
    stepsizemat.reshape(ssm_xy_nelem,2)[ssm_nan_dx,0]=dx
    ssm_nan_dy=np.isnan(stepsizemat.reshape(ssm_xy_nelem,2)[:,1])
    stepsizemat.reshape(ssm_xy_nelem,2)[ssm_nan_dx,1]=dy

    ssm_hires_xy_nelem=stepsizemat_hires.shape[0]*stepsizemat_hires.shape[1]
    ssm_hires_nan_dx=np.isnan(stepsizemat_hires.reshape(ssm_hires_xy_nelem,2)[:,0])
    stepsizemat_hires.reshape(ssm_hires_xy_nelem,2)[ssm_hires_nan_dx,0]=dx/hires_factor
    ssm_hires_nan_dy=np.isnan(stepsizemat_hires.reshape(ssm_hires_xy_nelem,2)[:,1])
    stepsizemat_hires.reshape(ssm_hires_xy_nelem,2)[ssm_hires_nan_dy,1]=dy/hires_factor
    
    
    minimal_curvature = maxabs_princcurvs < 1.0/(20*reflectors[0][0])
    nominal_scaling = ( (np.abs((stepsizemat[:,:,0]-dx)/dx) < 0.05) &  # less than 5% scaling error using nominal scaling factors
                    (np.abs((stepsizemat[:,:,1]-dy)/dy) < 0.05) )

    use_flat = minimal_curvature & nominal_scaling
    

    # scaled tikparam
    #raise ValueError("foo!")
    
    #z_reference=reflectors[-1][0]  # z coordinate of shallowest reflectors (recall reflectors are deepest first)
    #scaledtikparams=greensinversion.scale_tikparam(tikparam,z_reference,reflectors)

    #if tikparam is not None:
    #    # tikparam scaled diagnostic plot (multi-step)
    #    pl.figure(nextfignum)
    #    pl.clf()
    #    for inversioncnt in range(len(inversions)):
    #        pl.plot(inverses[inversioncnt][1] * (tikparam/scaledtikparams[inversioncnt])) #  * z_values[inversioncnt]/z_reference)
    #        pass
    #        pl.xlabel('Scaled singular value index')
    #        pl.ylabel('Magnitude')
    #        nextfignum+=1
    #    pass
    
    fullinverse=np.zeros((len(reflectors)+1,wfmdict[channel].data.shape[1]//xydownsample,wfmdict[channel].data.shape[0]//xydownsample),dtype='d')
    fullinverse_x_bnd=IniVal[0]-Step[0]*xydownsample/2.0 + np.arange(DimLen[0]//xydownsample+1,dtype='d')*Step[0]*xydownsample
    fullinverse_y_bnd=IniVal[1]-Step[1]*xydownsample/2.0 + np.arange(DimLen[1]//xydownsample+1,dtype='d')*Step[1]*xydownsample
    
    flat_tile = [ use_flat[yidx:(yidx+ny),xidx:(xidx+nx)].all() for (yidx,xidx) in minyminx_corners] 
    valid_tile = [ not((np.isnan(curvmat[yidx:(yidx+ny),xidx:(xidx+nx),:,:])).any()) for (yidx,xidx) in minyminx_corners ] 

    if channel_weights is None:
        # Assume all tiles have nonzero weights
        weighted_tile = [ True for (yidx,xidx) in minyminx_corners ]
        pass
    else: 
        # True only for tiles with a non-zero weight
        
        weights_data=wfmdict[channel_weights].data[::xydownsample,::xydownsample].T
        weighted_tile = [ (weights_data[yidx:(yidx+ny),xidx:(xidx+nx)] > 0.0).any()  for (yidx,xidx) in minyminx_corners ]
        
        pass



    #eval_linelength_avgcurvature_mirroredbox = lambda boxu1,boxv1,boxu2,boxv2,u1,v1,u2,v2: obj.implpart.surfaces[0].intrinsicparameterization.linelength_avgcurvature_mirroredbox_meshbased(obj.implpart.surfaces[0],curvmat_hires,stepsizemat_hires,obj.implpart.surfaces[0].intrinsicparameterization.lowerleft_meaningfulunits[0],obj.implpart.surfaces[0].intrinsicparameterization.lowerleft_meaningfulunits[1],curvmat_sizex*1.0/curvmat_hires.shape[1],curvmat_sizey*1.0/curvmat_hires.shape[0],boxu1,boxv1,boxu2,boxv2,dx,dy,u1,v1,u2,v2)
    #boxu1=-0.000447803
    #boxv1=-0.000447803
    #boxu2=0.021643787
    #boxv2=0.023434997
    #u1=-0.00014926799999999998
    #v1=-0.00014926799999999998
    #u2=-0.001343408
    #v2=-0.000746338
    #if np.isnan(eval_linelength_avgcurvature_mirroredbox(boxu1,boxv1,boxu2,boxv2,u1,v1,u2,v2)).any():
    #    raise ValueError("NAN")
    #break linelength_avgcurvature_mirroredbox_meshbased_c_one

    print("Defining flat surface inversion")
    # (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions,rowselects,inversions,inversionsfull,inverses,nresults)
    flat_inversion=define_flat_inversion(gi_params,flat_gi_grid,num_sources_y,num_sources_x)

    if do_singlestep_bool:
        print("Generating single-step inversion")

        # should be define_flat_inversion here  probably
        (ss_rowselects,ss_inversions,ss_inversionsfull,ss_inverses,ss_nresults)=greensinversion.generatesinglestepinversion(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,tstars,ny,nx,trange,depths)
        pass


    nextfignum=1

    # tikparam diagnostic plots (multi-step)
    if False:
        pl.figure(nextfignum)
        pl.clf()
        for inversioncnt in range(len(inversions)):
            pl.plot(inverses[inversioncnt][1])
            pass
        pl.xlabel('Singular value index')
        pl.ylabel('Magnitude')
        nextfignum+=1
        
        if do_singlestep_bool:
            pl.figure(nextfignum)
            pl.clf()
            pl.plot(ss_inverses[0][1])
            pl.xlabel('Singular value index (single step)')
            pl.ylabel('Magnitude')
            nextfignum+=1
            pass
        pass


    print("Iterating over %d tiles" % (len(minyminx_corners)))

    for tile_idx in range(len(minyminx_corners)): 
        (yidx,xidx)=minyminx_corners[tile_idx]
        
        print("Tile %d/%d" % (tile_idx,len(minyminx_corners)))

        #if tile_idx==27:
        #    raise ValueError("FOO!")

        inputmat = inputmats_holesfilled[tile_idx]

        if (flat_tile[tile_idx]) or not valid_tile[tile_idx] or not weighted_tile[tile_idx]: #  or tile idx > 75 or tile_idx != 27 # or tile_idx < 26 or tile_idx > 28
            (ny,nx,y,x,ygrid,xgrid,y_bnd,x_bnd) = flat_gi_grid
            (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions,rowselects,inversions,inversionsfull,inverses,nresults) = flat_inversion
            print("inverting with flat_inversion")
            pass
        elif valid_tile[tile_idx]:
            # build grid at this location
            gi_grid = build_gi_grid(dy,maxy,
                                    dx,maxx,
                                    firstcentery=IniVal[1]+yidx*dy,
                                    firstcenterx=IniVal[0]+xidx*dx)

            (ny,nx,y,x,ygrid,xgrid,y_bnd,x_bnd) = gi_grid

            try: 
                (rowscaling,
                 flashsourcecolumnscaling,
                 flashsourcevecs,
                 reflectorcolumnscaling,
                 reflectorsourcevecs,
                 depths,
                 tstars,
                 conditions,
                 prevconditions,
                 prevscaledconditions,
                 rowselects,
                 inversions,
                 inversionsfull,
                 inverses,
                 nresults) = define_curved_inversion(
                     gi_params,
                     gi_grid,
                     obj,
                     curvmat[yidx:(yidx+ny),xidx:(xidx+nx)],
                     stepsizemat[yidx:(yidx+ny),xidx:(xidx+nx)],
                     curvmat_hires,
                     stepsizemat_hires,
                     curvmat_sizex,
                     curvmat_sizey,
                     num_sources_y=num_sources_y,
                     num_sources_x=num_sources_x)
                pass
            except NotANumberError as e:
                sys.stderr.write("WARNING: Found NAN in sourcevecs... using flat (tile idx %d; yidx=%d, xidx=%d): %s\n" % (tile_idx,yidx,xidx,str(e)))
                (ny,nx,y,x,ygrid,xgrid,y_bnd,x_bnd) = flat_gi_grid
                (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions,rowselects,inversions,inversionsfull,inverses,nresults) = flat_inversion
                #raise #!!!
                pass
            
            print("inverting with curved_inversion")
            pass
        else:
            continue


        (inversioncoeffs_list,errs_list,tikparams_list) = inversionevalfunc(OpenCL_CTX,
                                                                            rowselects,
                                                                            inversions,
                                                                            inversionsfull,
                                                                            inverses,
                                                                            nresults,
                                                                            [ inputmat ],
                                                                            tikparam)
        
        
        fullinverse[:,yidx:(yidx+ny),xidx:(xidx+nx)] += greensinversion.buildconcreteinverse(inversioncoeffs_list[0],reflectors,ygrid,xgrid,y_bnd,x_bnd,ny,nx,num_sources_y,num_sources_x)*contributionprofiles[tile_idx]
    
        pass

    # raise ValueError("Debugging!")

    print("Performing assumed-flat inversion")
    print("Iterating over %d tiles" % (len(minyminx_corners)))
    flatfullinverse=np.zeros((len(reflectors)+1,wfmdict[channel].data.shape[1]//xydownsample,wfmdict[channel].data.shape[0]//xydownsample),dtype='d')

    for tile_idx in range(len(minyminx_corners)): 
        (yidx,xidx)=minyminx_corners[tile_idx]
        print("Tile %d/%d" % (tile_idx,len(minyminx_corners)))
        
        inputmat = inputmats_holesfilled[tile_idx]

        (ny,nx,y,x,ygrid,xgrid,y_bnd,x_bnd) = flat_gi_grid
        (rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions,rowselects,inversions,inversionsfull,inverses,nresults) = flat_inversion

        (inversioncoeffs_list,errs_list,tikparams_list) = inversionevalfunc(OpenCL_CTX,
                                                                            rowselects,
                                                                            inversions,
                                                                            inversionsfull,
                                                                            inverses,
                                                                            nresults,
                                                                            [ inputmat ],
                                                                            tikparam)
        flatfullinverse[:,yidx:(yidx+ny),xidx:(xidx+nx)] += greensinversion.buildconcreteinverse(inversioncoeffs_list[0],reflectors,ygrid,xgrid,y_bnd,x_bnd,ny,nx,num_sources_y=num_sources_y,num_sources_x=num_sources_x)*contributionprofiles[tile_idx]
        pass
    
    
    if do_singlestep_bool:

        (ss_inversioncoeffs_list,ss_errs_list,ss_tikparams_list) = inversionevalfunc(OpenCL_CTX,
                                                                                     ss_rowselects,
                                                                                     ss_inversions,
                                                                                     ss_inversionsfull,
                                                                                     ss_inverses,
                                                                                     ss_nresults,
                                                                                     inputmats_holesfilled,
                                                                                     tikparam)
        

        ss_fullinverse=np.zeros((len(reflectors)+1,wfmdict[channel].data.shape[1]//xydownsample,wfmdict[channel].data.shape[0]//xydownsample),dtype='d')

        for tile_idx in range(len(minyminx_corners)):
            (yidx,xidx)=minyminx_corners[tile_idx]
        
            ss_fullinverse[:,yidx:(yidx+ny),xidx:(xidx+nx)] += greensinversion.buildconcreteinverse(ss_inversioncoeffs_list[tile_idx],reflectors,ygrid,xgrid,y_bnd,x_bnd,ny,nx,num_sources_y,num_sources_x)*contributionprofiles[tile_idx]
            pass

        # for tile_idx in range(len(minyminx_corners)):
        #    (yidx,xidx)=minyminx_corners[tile_idx]
        #    #
        #    (ss_inversioncoeffs,ss_residual,errs,ss_tikparams)=greensinversion.performinversionsteps(ss_rowselects,ss_inversions,ss_inversionsfull,ss_inverses,ss_nresults,wfmdict[channel].data[(xidx*xydownsample):((xidx+nx)*xydownsample):xydownsample,(yidx*xydownsample):((yidx+ny)*xydownsample):xydownsample,startframe:endframe].transpose((2,1,0)),tikparam) # transpose to convert dataguzzler axis ordering (x,y,t) to greensinversion ordering (t,y,x)
        #    #
        #    ss_concreteinverse=greensinversion.buildconcreteinverse(ss_inversioncoeffs,reflectors,ygrid,xgrid,y_bnd,x_bnd,ny,nx)
        #    # concreteinverse is (len(reflectors)+1,ny,nx)... first layer is surface
        #    # ... accumulate contributions of each tile to full inverse
        #    ss_fullinverse[:,yidx:(yidx+ny),xidx:(xidx+nx)] += ss_concreteinverse*contributionprofiles[tile_idx]
        #    pass
        pass
        
    if tikparam is None:
        outpng_fname="%s_greensinversion.png" % (inputfile_basename)
        movieoutdirname="%s_greensinversion_movie/" % (inputfile_basename)
        movieoutfilename="%s_greensinversion_movie_depth_%%05.2f.png" % (inputfile_basename)
        outpngflat_fname="%s_greensinversionflat.png" % (inputfile_basename)
        movieoutflatfilename="%s_greensinversionflat_movie_depth_%%05.2f.png" % (inputfile_basename)
        pass
    else:
        outpng_fname="%s_greensinversion_tik_%g.png" % (inputfile_basename,tikparam)
        movieoutdirname="%s_greensinversion_tik_%g_movie/" % (inputfile_basename,tikparam)
        movieoutfilename="%s_greensinversion_tik_%g_movie_depth_%%05.2f.png" % (inputfile_basename,tikparam)
        outpngflat_fname="%s_greensinversionflat_tik_%g.png" % (inputfile_basename,tikparam)
        movieoutflatfilename="%s_greensinversionflat_tik_%g_movie_depth_%%05.2f.png" % (inputfile_basename,tikparam)
        pass

    (fig,subplots,images)=greensinversion.plotconcreteinverse(nextfignum,
                                                              dc_numplotrows_int,dc_numplotcols_int,
                                                              saturation_map,
                                                              fullinverse,
                                                              reflectors,-10000.0,30000.0,fullinverse_y_bnd,fullinverse_x_bnd,num_sources_y,num_sources_x)
    nextfignum+=1

    outpng_href=dc_value.hrefvalue(quote(outpng_fname),_dest_href)
    fig.savefig(outpng_href.getpath())
    reslist.append( (("dc:greensinversion_figure",{ "tikparam": str(tikparam)}), outpng_href))

    (fig,subplots,images)=greensinversion.plotconcreteinverse(nextfignum,
                                                              dc_numplotrows_int,dc_numplotcols_int,
                                                              saturation_map,
                                                              flatfullinverse,
                                                              reflectors,-10000.0,30000.0,fullinverse_y_bnd,fullinverse_x_bnd,num_sources_y,num_sources_x)
    nextfignum+=1

    outpngflat_href=dc_value.hrefvalue(quote(outpngflat_fname),_dest_href)
    fig.savefig(outpngflat_href.getpath())
    reslist.append( (("dc:greensinversion_figure",{ "tikparam": str(tikparam)}), outpngflat_href))
    

    
    movieoutdirhref=dc_value.hrefvalue(quote(movieoutdirname),contexthref=_dest_href)
    
    (nextfignum,plots,images,plothrefs,depths) = greensinversion.inversion.plotconcreteinversemovie(nextfignum,movieoutdirhref,movieoutfilename,saturation_map,fullinverse,reflectors,-10000.0,30000.0,fullinverse_y_bnd,fullinverse_x_bnd,num_sources_y,num_sources_x,dpi=300)

    for cnt in range(len(plothrefs)):
        reslist.append( (("dc:greensinversion_movie_frame",{ "tikparam": str(tikparam),"depth":str(depths[cnt])}), plothrefs[cnt]))
        pass

    (nextfignum,plots,images,plotflathrefs,depths) = greensinversion.inversion.plotconcreteinversemovie(nextfignum,movieoutdirhref,movieoutflatfilename,saturation_map,flatfullinverse,reflectors,-10000.0,30000.0,fullinverse_y_bnd,fullinverse_x_bnd,num_sources_y,num_sources_x,dpi=300)

    for cnt in range(len(plotflathrefs)):
        reslist.append( (("dc:greensinversionflat_movie_frame",{ "tikparam": str(tikparam),"depth":str(depths[cnt])}), plotflathrefs[cnt]))
        pass
    




    
    if do_singlestep_bool:
        (ss_fig,ss_subplots,ss_images)=greensinversion.plotconcreteinverse(nextfignum,dc_numplotrows_int,dc_numplotcols_int,saturation_map,ss_fullinverse,reflectors,-10000.0,30000.0,fullinverse_y_bnd,fullinverse_x_bnd,num_sources_y,num_sources_x)
        nextfignum+=1

        if tikparam is None:
            ss_outpng_fname="%s_ss_greensinversion.png" % (inputfile_basename)
            ss_movieoutdirname="%s_ss_greensinversion_movie/" % (inputfile_basename)
            ss_movieoutfilename="%s_ss_greensinversion_movie_depth_%%05.2f.png" % (inputfile_basename)
            pass
        else: 
            ss_outpng_fname="%s_ss_greensinversion_tik_%g.png" % (inputfile_basename,tikparam)
            ss_movieoutdirname="%s_ss_greensinversion_tik_%g_movie/" % (inputfile_basename,tikparam)
            ss_movieoutfilename="%s_ss_greensinversion_tik_%g_movie_depth_%%05.2f.png" % (inputfile_basename,tikparam)
            pass
        ss_outpng_href=dc_value.hrefvalue(quote(ss_outpng_fname),_dest_href)
        ss_fig.savefig(ss_outpng_href.getpath())
        reslist.append( (("dc:greensinversion_singlestep_figure", {"tikparam": str(tikparam) }), ss_outpng_href) )

        ss_movieoutdirhref=dc_value.hrefvalue(quote(ss_movieoutdirname),contexthref=_dest_href)

        (nextfignum,ss_plots,ss_images,ss_plothrefs,ss_depths) = greensinversion.inversion.plotconcreteinversemovie(nextfignum,ss_movieoutdirhref,ss_movieoutfilename,saturation_map,ss_fullinverse,reflectors,-10000.0,30000.0,fullinverse_y_bnd,fullinverse_x_bnd,num_sources_y,num_sources_x,resolution=300)

        for cnt in range(len(ss_plothrefs)):
            reslist.append( (("dc:ss_greensinversion_movie_frame",{ "tikparam": str(tikparam),"depth":str(ss_depths[cnt])}), ss_plothrefs[cnt]))
            pass
            

        pass
    
        
    outwfmdict={}

    outwfmdict[dc_cadmodel_channel_str]=copy.deepcopy(wfmdict[dc_cadmodel_channel_str])
    SplitTextureChans=dgm.GetMetaDatumWIStr(wfmdict[dc_cadmodel_channel_str],"TextureChans","").split("|")
    PrefixedTextureChans="|".join([ dc_prefix_str + TexChanPrefix + TexChan for TexChan in SplitTextureChans ])
    PrefixedFlatTextureChans="|".join([ dc_prefix_str + "_flat" + TexChanPrefix + TexChan for TexChan in SplitTextureChans ])

    gi_3d=dg.wfminfo()
    #gi_3d.Name=dc_prefix_str+dc_cadmodel_channel_str
    gi_3d.Name="Proj"+dc_prefix_str+TexChanPrefix+dc_cadmodel_channel_str
    gi_3d.dimlen=np.array((1,),dtype='i8')
    gi_3d.data=np.array((1,),dtype='f')
    dgm.AddMetaDatumWI(gi_3d,dgm.MetaDatum("VRML97GeomRef",dc_cadmodel_channel_str))
    dgm.AddMetaDatumWI(gi_3d,dgm.MetaDatum("X3DGeomRef",dc_cadmodel_channel_str))
    #texchanprefix=gi_3d.Name[:gi_3d.Name.find(dc_unprefixed_texname_str)]
    dgm.AddMetaDatumWI(gi_3d,dgm.MetaDatum("TexChanPrefix",dc_prefix_str+TexChanPrefix))
    dgm.AddMetaDatumWI(gi_3d,dgm.MetaDatum("TextureChans",PrefixedTextureChans))
    outwfmdict[gi_3d.Name]=gi_3d


    giflat_3d=dg.wfminfo()
    #gi_3d.Name=dc_prefix_str+dc_cadmodel_channel_str
    giflat_3d.Name="Proj"+dc_prefix_str+"flat_"+TexChanPrefix+dc_cadmodel_channel_str
    giflat_3d.dimlen=np.array((1,),dtype='i8')
    giflat_3d.data=np.array((1,),dtype='f')
    dgm.AddMetaDatumWI(giflat_3d,dgm.MetaDatum("VRML97GeomRef",dc_cadmodel_channel_str))
    dgm.AddMetaDatumWI(giflat_3d,dgm.MetaDatum("X3DGeomRef",dc_cadmodel_channel_str))
    #texchanprefix=giflat_3d.Name[:gi_3d.Name.find(dc_unprefixed_texname_str)]
    dgm.AddMetaDatumWI(giflat_3d,dgm.MetaDatum("TexChanPrefix",dc_prefix_str+"_flat"+TexChanPrefix))
    dgm.AddMetaDatumWI(giflat_3d,dgm.MetaDatum("TextureChans",PrefixedFlatTextureChans))
    outwfmdict[giflat_3d.Name]=giflat_3d
    #outwfm_flat.Name=dc_prefix_str+dc_inversion_channel_str+"flat"
    

    
    outwfm=dg.wfminfo()
    #outwfm.Name="greensinversion"
    outwfm.Name=dc_prefix_str+dc_inversion_channel_str

    outwfmdict[outwfm.Name]=outwfm

    outwfm_flat=dg.wfminfo()
    #outwfm.Name="greensinversion"
    outwfm_flat.Name=dc_prefix_str+"_flat"+dc_inversion_channel_str

    outwfmdict[outwfm_flat.Name]=outwfm_flat

    
    # Shift IniVals according to xydownsample:
    # IniVal[0] is X coordinate of center of corner pixel of undownsampled image
    # IniVal[0] is X coordinate of center of corner pixel downsampled image
    # but that pixel is twice as big, so the corner of the image itself
    # has changed!
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("IniVal1",IniVal[0]))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("IniVal2",IniVal[1]))
    
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("Step1",XStepMeters*xydownsample))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("Step2",YStepMeters*xydownsample))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("Coord1",Coord[0]))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("Coord2",Coord[1]))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("Units1",Units[0]))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("Units2",Units[1]))
    
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("IniVal3",0.0))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("Step3",1.0))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("Coord3","Depth Index"))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("Units3","unitless"))

    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("AmplCoord","Heating intensity"))
    dgm.AddMetaDatumWI(outwfm,dgm.MetaDatum("AmplUnits","J/m^2"))

    # Copy landmark metadata
    LandmarkMD = [ MDName for MDName in list(wfmdict[channel].MetaData.keys()) if MDName.startswith("LANDMARK_") ]
    for LandmarkName in LandmarkMD:
        dgm.AddMetaDatumWI(outwfm,copy.deepcopy(wfmdict[channel].MetaData[LandmarkName]))
        pass

    outwfm_flat.MetaData=copy.deepcopy(outwfm.MetaData)
    
    if channel_weights is not None:
        #outwfm_weights=copy.deepcopy(wfmdict[channel_weights])#dg.wfminfo()
        #outwfm_weights.Name="greensinversion_weights"
        outwfm_weights=dg.wfminfo()
        outwfm_weights.Name=dc_prefix_str+dc_inversion_channel_str+"_weights"
        outwfm_weights.data=wfmdict[channel_weights].data[::xydownsample,::xydownsample]
        outwfm_weights.dimlen=np.array(outwfm_weights.data.shape)
        outwfm_weights.ndim=2
        outwfm_weights.MetaData=copy.deepcopy(outwfm.MetaData)
        dgm.AddMetaDatumWI(outwfm_weights,dgm.MetaDatum("AmplCoord","Weighting"))
        dgm.AddMetaDatumWI(outwfm_weights,dgm.MetaDatum("AmplUnits","Unitless"))
        
        
        outwfmdict[outwfm_weights.Name]=outwfm_weights
        pass
    

    if do_singlestep_bool:
        ss_outwfm=copy.deepcopy(outwfm)
        ss_outwfm.Name="ss_greensinversion"
        
        outwfmdict[ss_outwfm.Name]=ss_outwfm
        pass

    
    # dgs file is written in (X,Y,Z) fortran order, so we write
    # dimlen in reverse order and transpose the data
    outwfm.ndim=3
    outwfm.dimlen=np.array(fullinverse.shape[::-1])
    outwfm.data=fullinverse.transpose().astype(np.float32)
    outwfm.NeedData=False
    outwfm.NeedMetaData=False
    outwfm.HaveData=True
    outwfm.HaveMetaData=True

    outwfm_flat.ndim=3
    outwfm_flat.dimlen=np.array(flatfullinverse.shape[::-1])
    outwfm_flat.data=flatfullinverse.transpose().astype(np.float32)
    outwfm_flat.NeedData=False
    outwfm_flat.NeedMetaData=False
    outwfm_flat.HaveData=True
    outwfm_flat.HaveMetaData=True

    outwfm_saturationmap=dg.wfminfo()
    outwfm_saturationmap.Name="saturation_map"
    outwfmdict[outwfm_saturationmap.Name]=outwfm_saturationmap
    outwfm_saturationmap.dimlen=np.array(saturation_map.shape[::-1])
    outwfm_saturationmap.data=saturation_map.transpose().astype(np.float32)
    outwfm_saturationmap.ndim=outwfm_saturationmap.dimlen.shape[0]
    outwfm_saturationmap.NeedData=False
    outwfm_saturationmap.NeedMetaData=False
    outwfm_saturationmap.HaveData=True
    outwfm_saturationmap.HaveMetaData=True
    outwfm_saturationmap.MetaData=copy.deepcopy(outwfm.MetaData)
    
    if do_singlestep_bool:
        ss_outwfm.ndim=3
        ss_outwfm.dimlen=np.array(ss_fullinverse.shape[::-1])
        ss_outwfm.data=ss_fullinverse.transpose().astype(np.float32)
        ss_outwfm.NeedData=False
        ss_outwfm.NeedMetaData=False
        ss_outwfm.HaveData=True
        ss_outwfm.HaveMetaData=True
        pass


    if tikparam is None:
        outdgs_fname="%s_greensinversion.dgs" % (inputfile_basename)        
        pass
    else:
        outdgs_fname="%s_greensinversion_tik_%g.dgs" % (inputfile_basename,tikparam)
        pass
    outdgs_href=dc_value.hrefvalue(quote(outdgs_fname),_dest_href)
    dgf.savesnapshot(outdgs_href.getpath(),outwfmdict)

    reslist.append( (("dc:greensinversion_dgsfile",{"tikparam": str(tikparam)}), outdgs_href))
    
    if do_singlestep_bool:
        pass

    # 
    # greensconvolution_params.get_opencl_context()
    # tile_idx=14
    # (yidx,xidx)=minyminx_corners[tile_idx]
    # 
    # inputmats=[wfmdict[channel].data[(xidx*xydownsample):((xidx+nx)*xydownsample):xydownsample,(yidx*xydownsample):((yidx+ny)*xydownsample):xydownsample,startframe:endframe].transpose((2,1,0))]
    # greeninversion.inversion.parallelperforminversionsteps(greensconvolution_params.OpenCL_CTX,rowselects,inversions,inversionsfull,inverses,nresults,inputmats,None)    
    return reslist
