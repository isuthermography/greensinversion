
import numpy as np
from matplotlib import pyplot as pl

import greensinversion
import greensconvolution


# Simple function returning 0 curvature for everything
# and pythagorean theorem for line length
def eval_linelength_avgcurvature_mirroredbox(boxu1,boxv1,boxu2,boxv2,u1,v1,u2,v2):

    linelength=np.sqrt((u1-u2)**2.0 + (v1-v2)**2.0)
    shape=np.broadcast(u1,v1,u2,v2).shape

    avgcurvature=np.zeros(shape,dtype='d')
    avgcrosscurvature=np.zeros(shape,dtype='d')
    theta=np.zeros(shape,dtype='d')

    return (linelength,avgcurvature,avgcrosscurvature,theta)


greensconvolution_params=greensconvolution.greensconvolution_calc.read_greensconvolution()

dy=0.8e-3
dx=0.8e-3
maxy=23.e-3
maxx=21.e-3

(ny,nx,
 y,x,
 ygrid,xgrid,
 y_bnd,x_bnd)=greensinversion.build_gi_grid(dy,maxy,
                                            dx,maxx)

rho=float(1.555e3) # kg/m^3
c=float(850.0) # J/(kg* deg K)
alphaz=float(.62e-6)
alphaxy=float(2.87e-6)

stepsizemat_tile=np.zeros((ny,nx,2),dtype='d')
stepsizemat_tile[:,:,0]=dx
stepsizemat_tile[:,:,1]=dy

kz=alphaz*rho*c
ky=alphaxy*rho*c
kx=alphaxy*rho*c

t0=10e-3
dt = 10e-3
nt=300

trange=t0+np.arange(nt,dtype='d')*dt
rowscaling=1.0

num_sources_y=11
num_sources_x=10

(flashsourcecolumnscaling,flashsourcevecs)=greensinversion.sourcevecs.build_flash_source_vecs_curved(greensconvolution_params,y,x,ygrid,xgrid,y_bnd,x_bnd,stepsizemat_tile,rho,c,kz,ky,kx,trange,rowscaling,eval_linelength_avgcurvature_mirroredbox,num_sources_y,num_sources_x)        

(flashsourcecolumnscaling_flat,flashsourcevecs_flat)=greensinversion.sourcevecs.build_flash_source_vecs(ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,trange,rowscaling,num_sources_y,num_sources_x)

# Look at a central sourcevec
vecnum=num_sources_y*num_sources_x//2

curved=flashsourcevecs.reshape(nt,ny,nx,num_sources_y*num_sources_x)[:,:,:,vecnum]
flat=flashsourcevecs_flat.reshape(nt,ny,nx,num_sources_y*num_sources_x)[:,:,:,vecnum]

pl.figure(1)
pl.clf()
pl.subplot(2,1,1)
pl.imshow(curved[0,:,:])
pl.colorbar()

pl.subplot(2,1,2)
pl.imshow(flat[0,:,:])
pl.colorbar()


pl.figure(2)
pl.clf()
pl.loglog(trange,curved[:,ny//2,nx//2],'-',
          trange,flat[:,ny//2,nx//2],'--')
pl.loglog(trange,curved[:,ny//2+1,nx//2],'-',
          trange,flat[:,ny//2+1,nx//2],'--')
pl.grid()

# Look at all sourcevecs
curved_all=np.sum(flashsourcevecs*flashsourcecolumnscaling,1).reshape(nt,ny,nx)
flat_all=np.sum(flashsourcevecs_flat*flashsourcecolumnscaling_flat,1).reshape(nt,ny,nx)


pl.figure(3)
pl.clf()
pl.subplot(2,1,1)
pl.loglog(trange,curved_all[:,ny//2,nx//2])
pl.axis((.001,10,1e-4,1e-1))
pl.grid()

pl.subplot(2,1,2)
pl.loglog(trange,flat_all[:,ny//2,nx//2])
pl.axis((.001,10,1e-4,1e-1))
pl.grid()


# compare ratio of sum of sourcevecs from curved to flat case
# (should roughly be 1.0)
ratio = curved_all[:,ny//2,nx//2]/flat_all[:,ny//2,nx//2]

pl.figure(4)
pl.clf()
pl.semilogx(trange,ratio,'-')

# Ratio should be close to 1 everywhere
assert(((ratio > .98) & (ratio < 1.01)).all())

# Look at reflectorvecs

reflector_depth=.5e-3
reflector_ny = 25
reflector_nx = 21

gc_kernel="opencl_interpolator"
gc_kernel_curved="opencl_interpolator_curved"
sourceveccolumnscaling = ((y_bnd[-1]-y_bnd[0])/reflector_ny) * ((x_bnd[-1]-x_bnd[0])/reflector_nx)/(reflector_depth**2.0)
sourcevecscalefactor=rowscaling/sourceveccolumnscaling

flatsourcevecs=greensinversion.build_reflector_source_vecs(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,rho,c,kz,ky,kx,trange,reflector_depth,reflector_ny,reflector_nx,gc_kernel)*sourcevecscalefactor

curvmat_tile=np.zeros((ny,nx,2,2),dtype='d')
stepsizemat_tile=np.zeros((ny,nx,2),dtype='d')
stepsizemat_tile[:,:,0]=dx
stepsizemat_tile[:,:,1]=dy


curvedsourcevecs=greensinversion.sourcevecs.build_reflector_source_vecs_curved(greensconvolution_params,dy,dx,ygrid,xgrid,y_bnd,x_bnd,stepsizemat_tile,curvmat_tile,rho,c,kz,ky,kx,trange,reflector_depth,reflector_ny,reflector_nx,gc_kernel_curved,eval_linelength_avgcurvature_mirroredbox)*sourcevecscalefactor


# These plots should match
flat_center_center=flatsourcevecs.reshape(nt,ny,nx,reflector_ny,reflector_nx)[:,ny//2,nx//2,reflector_ny//2,reflector_nx//2]
curved_center_center=curvedsourcevecs.reshape(nt,ny,nx,reflector_ny,reflector_nx)[:,ny//2,nx//2,reflector_ny//2,reflector_nx//2]

pl.figure(5)
pl.clf()
pl.plot(trange,flat_center_center,'-',
        trange,curved_center_center,'-')

assert(np.linalg.norm(flat_center_center-curved_center_center)/np.linalg.norm(flat_center_center) < 1e-6)

(reflector_widthy,
 reflector_widthx,
 reflector_posy,
 reflector_posx,
 reflector_ygrid,
 reflector_xgrid,
 reflector_bndy,
 reflector_bndx)=greensinversion.sourcevecs.definereflectors(y_bnd,x_bnd,reflector_ny,reflector_nx)


flat_gf = greensconvolution.greensconvolution_fast.greensconvolution_integrate_anisotropic(greensconvolution_params,np.array((reflector_depth,),dtype='f'),np.array((0,),dtype='f'),trange.astype(np.float32),0.0,kz,ky,kx,rho,c,reflector_widthy*reflector_widthx*2.0, (),kernel=gc_kernel,opencl_queue=None)

refl_yidx=reflector_ny//2
refl_xidx=reflector_nx//2

reflector_pixels_y = (y >= (reflector_posy[refl_yidx]-reflector_widthy/2.0)) & (y <= (reflector_posy[refl_yidx]+reflector_widthy/2.0))
reflector_pixels_x = (x >= (reflector_posx[refl_xidx]-reflector_widthx/2.0)) & (x <= (reflector_posx[refl_xidx]+reflector_widthx/2.0))
reflector_pixels_widthy = stepsizemat_tile[reflector_pixels_y,:,:][:,reflector_pixels_x,1]
reflector_pixels_widthx = stepsizemat_tile[reflector_pixels_y,:,:][:,reflector_pixels_x,0]
reflector_pixels_area = np.sum(reflector_pixels_widthy*reflector_pixels_widthx) # sum (dx*dy) over both axes to get total area 


reflector_pixels_nominalarea = dy*np.count_nonzero(reflector_pixels_y) * dx*np.count_nonzero(reflector_pixels_x)

surface_reflector_area = reflector_pixels_area * (reflector_widthy*reflector_widthx)/reflector_pixels_nominalarea

extra_area_factor=0.0
reflector_area = (1.0+extra_area_factor)*surface_reflector_area

curved_gf = greensconvolution.greensconvolution_fast.greensconvolution_integrate_anisotropic(greensconvolution_params,np.array((reflector_depth,),dtype='f'),np.array((0,),dtype='f'),trange.astype(np.float32),0.0,kz,ky,kx,rho,c,reflector_area*2.0, (),avgcurvatures=np.array((0,),dtype='f'),kernel=gc_kernel_curved)

# These plots should match also
pl.figure(6)
pl.clf()
pl.plot(trange,flat_gf,'-',
        trange,curved_gf,'-')

assert(np.linalg.norm(flat_gf-curved_gf)/np.linalg.norm(flat_gf) < 1e-6)

pl.show()
