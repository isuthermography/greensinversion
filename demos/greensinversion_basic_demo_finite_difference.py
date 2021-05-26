import sys
import os

import numpy as np
from matplotlib import pyplot as pl
import heatsim2

# !!! Should add back layer (infinite or perhaps quartered) sheet at assumed thickness) which 
# we can solve for the intensity of. 
#  * Or perhaps we allow an infinite sheet at every layer?
#  * Either would help to remove the lousy results at the sample back wall
#  * Better yet have each border element be integrated over twice its
#    area off-screen (IMPLEMENTED THIS)
#


from greensconvolution.greensconvolution_calc import read_greensconvolution


import greensinversion
from greensinversion.heatsim_calc import heatsim_calc

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
dy=1e-3
dx=1e-3

nz=16

(ny,nx,
 z,y,x,
 zgrid,ygrid,xgrid,
 z_bnd,y_bnd,x_bnd) = greensinversion.build_gi_grid_3d(dz,nz,
                                                    dy,38.0e-3,
                                                    dx,36.0e-3)


# number of top-layer sources in each tile
num_sources_y=2
num_sources_x=2 


#t0=0.01
# t0 bumped up because finite difference doesn't work too well
# at short times
t0=.07
dt=1.0/30.0
# nt=250
nt=1800

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


(rowscaling,flashsourcecolumnscaling,flashsourcevecs,reflectorcolumnscaling,reflectorsourcevecs,depths,tstars,conditions,prevconditions,prevscaledconditions)=greensinversion.build_all_source_vecs(greensconvolution_params,dy,dx,ygrid[0,:,:],xgrid[0,:,:],y_bnd,x_bnd,rho,c,kz,ky,kx,dt,trange,reflectors,num_sources_y=num_sources_y,num_sources_x=num_sources_x)





#depth1=z_bnd[13]
#reflectorsourcevecs1=build_reflector_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,trange,depth1,4,4)
#(u1,s1,v1)=np.linalg.svd(reflectorsourcevecs1,full_matrices=False)

#print("s1 ratio=%f" % (max(s1)/min(s1)))

#print("conditions=%s" % (str(conditions)))
#print("prevconditions=%s" % (str(prevconditions)))
#print("prevscaledconditions=%s" % (str(prevscaledconditions)))


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

tikparam=2e-10

# run the simulation 
hole_min_z=0.8e-3 
#hole_min_z=320*dz
#hole_min_z=z_thick 
hole_min_x=3e-3
hole_max_x=20e-3
hole_min_y=12.e-3
hole_max_y=24.e-3

thininsulatinglayerdepth=2*dz
#thininsulatinglayerdepth=320*dz
thininsulatinglayer_min_x=16e-3
thininsulatinglayer_max_x=32e-3
thininsulatinglayer_min_y=15e-3
thininsulatinglayer_max_y=30e-3
    

T=heatsim_calc(ny,nx,dz,dx,dy,
               z,y,x,
               zgrid,ygrid,xgrid,
               z_bnd,y_bnd,x_bnd,
               kz,ky,kx,rho,c,
               trange,dt,
               hole_min_z,
               hole_min_y,
               hole_max_y,
               hole_min_x,
               hole_max_x,
               thininsulatinglayerdepth,
               thininsulatinglayer_min_y,
               thininsulatinglayer_max_y,
               thininsulatinglayer_min_x,
               thininsulatinglayer_max_x)

Tnoisy=T+np.random.randn(*T.shape)*.022 # 22 mK NETD

# To plot: 
# loglog(trange+dt/2,T[20,20,:])
# imshow(T[:,:,200]

# Saturation check not technically necessary on simulated data 
(saturation_fraction,saturation_map)=greensinversion.saturationcheck(Tnoisy,0)


(inversioncoeffs,residual,errs,tikparams)=greensinversion.performinversionsteps(rowselects,inversions,inversionsfull,inverses,nresults,Tnoisy,tikparam)

(fig,subplots,images)=greensinversion.plotabstractinverse(5,2,3,inversioncoeffs,reflectors,-10000.0,20000.0,y_bnd,x_bnd,num_sources_y,num_sources_x)

concreteinverse=greensinversion.buildconcreteinverse(inversioncoeffs,reflectors,ygrid[0,:,:],xgrid[0,:,:],y_bnd,x_bnd,ny,nx,num_sources_y,num_sources_x)
(cfig,csubplots,cimages)=greensinversion.plotconcreteinverse(6,2,3,saturation_map,concreteinverse,reflectors,-10000.0,20000.0,y_bnd,x_bnd,num_sources_y,num_sources_x)


(ss_inversioncoeffs,ss_residual,ss_errs,ss_tikparams)=greensinversion.performinversionsteps(ss_rowselects,ss_inversions,ss_inversionsfull,ss_inverses,ss_nresults,Tnoisy,tikparam)

(ss_fig,ss_subplots,ss_images)=greensinversion.plotabstractinverse(7,2,3,ss_inversioncoeffs,reflectors,-10000.0,20000.0,y_bnd,x_bnd,num_sources_y,num_sources_x)
pl.savefig("/tmp/greensinversion_basic_singlestep_thininsulatinglayer_right_deephole_left_abstract.png")

ss_concreteinverse=greensinversion.buildconcreteinverse(ss_inversioncoeffs,reflectors,ygrid[0,:,:],xgrid[0,:,:],y_bnd,x_bnd,ny,nx,num_sources_y,num_sources_x)

(ss_cfig,ss_csubplots,ss_cimages)=greensinversion.plotconcreteinverse(8,2,3,saturation_map,ss_concreteinverse,reflectors,-10000.0,20000.0,y_bnd,x_bnd,num_sources_y,num_sources_x)


pl.show()
