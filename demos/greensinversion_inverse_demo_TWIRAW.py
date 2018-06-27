import numpy as np
from greensinversion.TWI_ReadRAW import ReadRAW
from greensinversion import perform_flat_inversion

# NOTE: Be careful that all parameters that should 
# be real numbers are floating point (i.e. with a decimal point)

# Since the TWI files are not temperature calibrated, 
# your Tikhonov parameter will have to be scaled 
# with the camera noise. 
# I use 7.5e-11 for a camera with an NETD of 22mK
# So if you know your camera NETD in detector counts, 
# try tikparam = 7.5e-11*NETD_in_detector_counts/22e-3
# as a starting point, or use the diagnostic plots
# to find the elbow in the magnitudes of the singular values
# and cut off somewhere above that point

tikparam = 5e-10
filename = "60 hz 907 frames flat bottom hole standard.RAW"

# Note that really only the product of rho and c matters
rho=1.555e3 # kg/m^3
c=850.0 # J/(kg*deg. K) 

# through thickness diffusivity
alphaz=.62e-6
# in-plane diffusivity
alphaxy=2.87e-6


# x and y initial positions -- only used for plotting
x0=0.0
y0=0.0

# Spatial pixel size must be measured from the thermal image sequence
dx=100.0e-3/244  # pixel size, in mm
dy=100.0e-3/244

# Apply spatial downsampling to keep inversion complexity under control
# This should be an integer
xydownsample=1

# Size of tiles to be used in the inversion process
tile_size_x=22.0e-3
tile_size_y=24.0e-3

# Set OpenCL device parameters to None or values found 
# as "Device Type" and "Device Name" with the "clinfo" command
OpenCL_Device_Type="GPU"  
OpenCL_Device_Name=None # e.g. "Quadro GP100"

singlestep=False

parallelevaluate=False   # Parallelevaluate uses OpenCL kernel to evaluate inversion, rather than CPU. Based on current experience, GPU is currently slightly SLOWER here (WHY?) so we don't use it



# reflectors is array of reflector geometries for each tile
# deepest first,
# e.g. reflectors = (  (deepest_depth, ny, nx),
#                      (second_deepest_depth, ny, nx),
#                       ...
#                      (second_shallowest_depth, ny, nx),
#                      (shallowest_depth, ny, nx))
nominal_lamina_thickness=8.05e-3/(3.0*16.0)   # (meters)

reflectors=( (nominal_lamina_thickness*16,4,4),
             (nominal_lamina_thickness*13,5,5),
             (nominal_lamina_thickness*10,6,6),
             (nominal_lamina_thickness*8,7,7),
             (nominal_lamina_thickness*7,8,8),
             (nominal_lamina_thickness*6,9,9),
             (nominal_lamina_thickness*5,11,11),
             (nominal_lamina_thickness*4,14,14),
             (nominal_lamina_thickness*3,18,18),
             (nominal_lamina_thickness*2,28,28))
#(nominal_lamina_thickness*1,28,28))


# number of rows and number of columns for figure full
# of subplots. numplotrows*numplotcols >= len(reflectors)+2
numplotrows=3
numplotcols=4 

# Source grid for flash energy representation.
# Note that these sources overlap each other so are
# twice as big in each axis as you would expect
source_approx_dy=2e-3
source_approx_dx=2e-3

plot_min_power_per_area = -1000000.0 # lower colormap bound on image plots
plot_max_power_per_area = 2300000.0 # upper colormap bound on image plots

# nominal_lamina_thickness
frames_to_discard = 4 # discard this many frames including/after the flash

frames_to_discard_prior = 1 # discard this many frames before the flash
# from the calculation used for background subtraction

nextfignum=1 # set to None to disable figure generation, set to 1 to enable figure generation


(data_t0,dt,flashframe,HeaderParams,data)=ReadRAW(filename)
# data_t0 is time of first frame in the data sequence (arbitrary reference)

flashtime=data_t0+flashframe*dt
# You can mask out invalid data such as off the edge of the 
# sample by setting it to NaN. This is not restricted 
# to rectangular cuts 
# (the missing region is interpolated from the valid
# data prior to inversion)

data[:,:110,:]=np.NaN
data[:,340:,:]=np.NaN
data[:,:,:205]=np.NaN
data[:,:,445:]=np.NaN


(inversionprob,
 saturation_map,
 inversion_info,
 fullinverse,
 nextfignum) = perform_flat_inversion(rho,c,
                                      alphaz,alphaxy,
                                      y0,x0,
                                      dy,dx,
                                      tile_size_y,tile_size_x,
                                      xydownsample,
                                      reflectors,
                                      source_approx_dy,
                                      source_approx_dx,
                                      tikparam,
                                      data_t0,dt,
                                      flashframe,flashtime,
                                      frames_to_discard,
                                      frames_to_discard_prior,
                                      data,
                                      singlestep,
                                      parallelevaluate,
                                      OpenCL_Device_Type,
                                      OpenCL_Device_Name,
                                      numplotrows,numplotcols,
                                      plot_min_power_per_area,
                                      plot_max_power_per_area,
                                      nextfignum)

# fullinverse is now a 3D array with the inverted intensities. 
# You can plot slices from it with pl.imshow()

# unwrap variables from result tuples
# (this is optional)
(minyminx_corners,
 yranges,
 xranges,
 contributionprofiles,
 inversioncoeffs_list,
 errs_list,
 tikparams_list,
 fullinverse_y_bnd,
 fullinverse_x_bnd) = inversion_info

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
# (this is optional)

# Show plots if we are configured to create them 
if nextfignum is not None:
    from matplotlib import pyplot as pl
    pl.show()
    pass
