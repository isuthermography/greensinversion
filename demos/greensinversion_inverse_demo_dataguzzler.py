import sys
import os

import dg_units
import dg_eval
import dg_file as dgf

import numpy as np
from matplotlib import pyplot as pl



from greensconvolution.greensconvolution_calc import read_greensconvolution

from limatix import dc_value


import greensinversion
#import greensinversion.tile_rectangle
#from greensinversion.heatsim_calc import heatsim_calc

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

rho=float(1.555e3) # kg/m^3
c=float(850.0) # J/(kg* deg K)
alphaz=float(.54e-6) # average value from measurements (Thermal_Properties.ods 11/25/15, averaging in-plane value from 90deg specimen and flash method values)
alphaxy=float(2.70e-6) # best guess based on Thermal_Properties.ods 11/25/15 based on 0/90 and quasi-isotropic layups
kx=alphaxy*rho*c
ky=alphaxy*rho*c
kz=alphaz*rho*c

# Lamina thickness based on thermal_properties.ods average thickness of 8.05 mm for 3(?) layers of 16 plies
lamina_thickness=8.05e-3/(3.0*16.0)

dz=lamina_thickness
#dy=.25e-3
#dx=.25e-3
dy=.5e-3
dx=.5e-3

nz=16

# Define a tile, 38x36 mm
(ny,nx,
 y,x,
 ygrid,xgrid,
 y_bnd,x_bnd) = greensinversion.build_gi_grid(dy,38.0e-3,
                                              dx,36.0e-3)

z_bnd=np.arange(nz+1,dtype='d')*dz  # z boundary starts at zero

# number of top-layer sources in each tile
num_sources_y=2
num_sources_x=2 



#t0=0.01
# t0 bumped up because finite difference doesn't work too well
# at short times
#t0=.07
t0=0.052869999999999973 # must match a frame time (relative to flash) of file we are inverting
#dt=1.0/30.0
dt=0.016684000000000001  # must match file we are inverting
# nt=250
nt=1835  # must match (or perhaps in the future be smaller than) the file we are inverting

trange=t0+np.arange(nt,dtype='d')*dt

# reflectors is a tuple of (z,ny,nx) tuples representing
# possible z values for reflectors and how many y and x pieces
# they should be split into.
# it should be ordered from the back surface towards the
# front surface. 

    # reflectors is (depth, reflector_ny,reflector_nx)
reflectors=( (z_bnd[15],4,4),
             (z_bnd[9],4,4),
             (z_bnd[5],6,6),
             (z_bnd[2],10,10))
             

greensconvolution_params=read_greensconvolution()




#reflectorsourcevecs=build_reflector_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,trange,
#                                                dz*20, # reflector_depth
#                                                3, # reflector_ny
#                                                3) # reflector_nx

#reflectordepth=dz*3
#
#reflector_ny=3
#reflector_nx=3
# can view individual source maps with
# reflectorsourcevecs[:,0].reshape(ny,nx,nt),
# e.g. imshow(reflectorsourcevecs[:,5].reshape(ny,nx,nt)[:,:,200])


# (u,s,v)=np.linalg.svd(reflectorsourcevecs,full_matrices=False)

# Noise gain limit
#  1111222233334444
#  1122334455667789
#  1234567890123456
# vs
#  111111222222333333
#  11
 

# Process for determining tile size/layout:
# Start with material thickness (or maximum depth of
# delamination of interest) L and thermal diffusivity
# Break time t* is L^2/(pi*alpha_z)
# Should record until at least 3*break time

# say 1/8", break time around 20 secs, record for a minute.

# Consider delam at deepest reasonable position, e.g.
# 2nd boundary up from bottom layer, with a 4x4 grid
# (z_bnd[-3]) 
#reflectorsourcevecs=build_reflector_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,trange,z_bnd[-3],4,4)
# then calc svd and look at max(s)/min(s)
# (u,s,v)=np.linalg.svd(reflectorsourcevecs,full_matrices=False)

# Adjust domain size until max(s)/min(s) is somewhere between 10 and 40
#
#
# Then consider a shallower depth, perhaps 2-4 ply boundaries up.
# Perform same calculation and svd
#reflectorsourcevecs2=build_reflector_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,trange,z_bnd[-7],4,4)
# (u2,s2,v2)=np.linalg.svd(reflectorsourcevecs2,full_matrices=False)
# if max(s2)/min(s2) < 10, increase grid size from 4x4 to 5x5 or 6x6
#
# Then evaluate the concatenation of both matrices
# (u12,s12,v12)=np.linalg.svd(np.concatenate((reflectorsourcevecs,reflectorsourcevecs2),axis=1),full_matrices=False)
# and make sure max(s12)/min(s12) is still < 40. If so may also check to
# make sure that there're shouldn't be a deeper layer. If not, need
# to go still shallower. 


print("Building source vecs")
(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions)=greensinversion.build_all_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,dt,trange,reflectors,num_sources_y=num_sources_y,num_sources_x=num_sources_x)



#depth1=z_bnd[13]
#reflectorsourcevecs1=build_reflector_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,trange,depth1,4,4)
#(u1,s1,v1)=np.linalg.svd(reflectorsourcevecs1,full_matrices=False)

#print("s1 ratio=%f" % (max(s1)/min(s1)))

print("conditions=%s" % (str(conditions)))
print("prevconditions=%s" % (str(prevconditions)))
print("prevscaledconditions=%s" % (str(prevscaledconditions)))


pl.figure(1)
pl.clf()
pl.imshow(reflectorsourcevecs[0][:,5].reshape(nt,ny,nx)[200,:,:])



pl.figure(2)
pl.clf()
pl.imshow(reflectorsourcevecs[1][:,5].reshape(nt,ny,nx)[200,:,:])

print("Generating inversion steps")

(rowselects,inversions,inversionsfull,inverses,nresults)=greensinversion.generateinversionsteps(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,tstars,ny,nx,trange,depths)

print("Generating single-step inversion")

(ss_rowselects,ss_inversions,ss_inversionsfull,ss_inverses,ss_nresults)=greensinversion.generatesinglestepinversion(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,tstars,ny,nx,trange,depths)

# diagnostic plots for finding Tikhonov parameter
pl.figure(3)
pl.clf()
for inversioncnt in range(len(inversions)):
    pl.plot(inverses[inversioncnt][1])
    pass
pl.xlabel('Singular value index')
pl.ylabel('Magnitude')

pl.figure(4)
pl.clf()
pl.plot(ss_inverses[0][1])
pl.xlabel('Singular value index (single step)')
pl.ylabel('Magnitude')

tikparam=1e-10

# To plot: 
# loglog(trange+dt/2,T[20,20,:])
# imshow(T[:,:,200]


# Load input file
# NOTE: When changing input file: 
#  1. Verify flashtime. Adjust as appropriate
#  2. Verify startframe. Adjust as appropriate
#  3. Execute file load code (below) and evaluate
#    a) XStepMeters (must match dx)
#    b) YStepMeters (must match dy)
#    c) TStep (must match dt)
#    d) bases[2][startframe]-flashtrigtime  (must match t0)
#    e) bases[2][startframe:].shape[0] (must match nt)
#  4. Adjust dx, dy, dt, t0, and/or nt to satisfy above criteria
#  5. Once adjusted, assert()s below should pass. 
inputfile="/tmp/CA-1_Bottom_2015_11_19_undistorted_orthographic.dgs"
flashtrigtime=2.70000 # seconds
flashtime=flashtrigtime+1.0/100.0 # add 1/100th second delay of flash peak (wild guess!)
channel="DiffStack"
# frame #165: Time relative to trigger = bases[2][165]-flashtrigtime
#                                      = 0.052869999999999973
startframe=165  # zero-based, not one-based

# Read in data file with thermal image sequence
(junkmd,wfmdict)=dgf.loadsnapshot(inputfile,memmapok=True)

# Extract spatial and temporal step sizes from file
(ndim,DimLen,IniVal,Step,bases)=dg_eval.geom(wfmdict[channel],raw=True)
(ndim,Coord,Units,AmplCoord,AmplUnits)=dg_eval.axes(wfmdict[channel],raw=True)
XIniValMeters=dc_value.numericunitsvalue(IniVal[0],Units[0]).value('m')
YIniValMeters=dc_value.numericunitsvalue(IniVal[1],Units[1]).value('m')
XStepMeters=dc_value.numericunitsvalue(Step[0],Units[0]).value('m')
YStepMeters=dc_value.numericunitsvalue(Step[1],Units[1]).value('m')
TStep=Step[2]
# step sizes from file must match inversion step sizes
assert(XStepMeters==dx)
assert(YStepMeters==dy)
assert(TStep==dt)
assert(bases[2][startframe]-flashtrigtime==t0)  # Start time matches
assert(bases[2][startframe:].shape[0]==nt) # Number of frames match

# Perform saturation check
(saturation_fraction,saturation_map)=greensinversion.saturationcheck(wfmdict[channel].data.transpose((2,1,0)),startframe)
    
if saturation_fraction > .2: 
    raise ValueError("greensinversion_inverse_demo_dataguzzler: ERROR: %.1f%% of pixels are saturated at least once beyond start frame!" % (saturation_fraction*100.0))
if saturation_fraction > .02:
    sys.stderr.write("greensinversion_inverse_demo_dataguzzler: WARNING: %.1f%% of pixels are saturated at least once beyond start frame!\n" % (saturation_fraction*100.0))
    pass


# Break image, stored in wfmdict[channel].data, into tiles of ny*nx pixels, perform inversion on each tile
(minyminx_corners,yranges,xranges,contributionprofiles)=greensinversion.build_tiled_rectangle(ny,nx,dy,dx,reflectors,wfmdict[channel].data.transpose((2,1,0)))
                                                                                              

# Create a data cube for storing the inversion
fullinverse=np.zeros((len(reflectors)+1,wfmdict[channel].data.shape[1],wfmdict[channel].data.shape[0]),dtype='d')


# Iterate through the tiles
for tile_idx in range(len(minyminx_corners)):
    (yidx,xidx)=minyminx_corners[tile_idx]

    # Perform the inversion on this tile   
    (inversioncoeffs,residual,errs,tikparams)=greensinversion.performinversionsteps(rowselects,inversions,inversionsfull,inverses,nresults,wfmdict[channel].data[xidx:(xidx+nx),yidx:(yidx+ny),startframe:].transpose((2,1,0)),tikparam) # transpose to convert dataguzzler axis ordering (x,y,t) to greensinversion ordering (t,y,x)

    # Build a concrete reconstruction of the buried heat sources from the inversion coefficients
    concreteinverse=greensinversion.buildconcreteinverse(inversioncoeffs,reflectors,ygrid,xgrid,y_bnd,x_bnd,ny,nx,num_sources_y,num_sources_x)
    # concreteinverse is (len(reflectors)+1,ny,nx)... first layer is surface

    # accumulate weighted contributions of this tile to full inverse
    fullinverse[:,yidx:(yidx+ny),xidx:(xidx+nx)] += concreteinverse*contributionprofiles[tile_idx]
    pass

# Create a data cube for storing the single-step inversion
ss_fullinverse=np.zeros((len(reflectors)+1,wfmdict[channel].data.shape[1],wfmdict[channel].data.shape[0]),dtype='d')

for tile_idx in range(len(minyminx_corners)):
    (yidx,xidx)=minyminx_corners[tile_idx]
    
    # Perform the single-step inversion on this tile
    (ss_inversioncoeffs,ss_residual,errs,ss_tikparams)=greensinversion.performinversionsteps(ss_rowselects,ss_inversions,ss_inversionsfull,ss_inverses,ss_nresults,wfmdict[channel].data[xidx:(xidx+nx),yidx:(yidx+ny),startframe:].transpose((2,1,0)),None) # transpose to convert dataguzzler axis ordering (x,y,t) to greensinversion ordering (t,y,x)

    # Build a concrete reconstruction of the buried heat sources from the inversion coefficients        
    ss_concreteinverse=greensinversion.buildconcreteinverse(ss_inversioncoeffs,reflectors,ygrid,xgrid,y_bnd,x_bnd,ny,nx,num_sources_y,num_sources_x)
    # concreteinverse is (len(reflectors)+1,ny,nx)... first layer is surface
    
    # ... accumulate contributions of each tile to full inverse
    ss_fullinverse[:,yidx:(yidx+ny),xidx:(xidx+nx)] += ss_concreteinverse*contributionprofiles[tile_idx]
    pass


(fig,subplots,images)=greensinversion.plotconcreteinverse(5,2,3,fullinverse,reflectors,-10000.0,30000.0,y_bnd,x_bnd,num_sources_y,num_sources_x)

(ss_fig,ss_subplots,ss_images)=greensinversion.plotconcreteinverse(6,2,3,ss_fullinverse,reflectors,-10000.0,30000.0,y_bnd,x_bnd,num_sources_y,num_sources_x)

pl.show()



