import numpy as np
import heatsim2



def heatsim_calc(ny,nx,dz,dx,dy,
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
                 thininsulatinglayer_max_x):
    flash_energy=10e3 # J/m^2
    #
    dz_refinement=3.0
    nz=z.shape[0]
    nz_refine=int(nz*dz_refinement)

    z_bnd_refine=np.arange(nz_refine+1,dtype='d')*dz/dz_refinement  # z boundary starts at zero
    z_refine=z_bnd_refine[:-1]+dz/(dz_refinement*2.0)

    (zgrid_refine,ygrid,xgrid) = np.meshgrid(z_refine,y,x,indexing='ij')

    #
    (z_bnd_z,z_bnd_y,z_bnd_x)=np.meshgrid(z_bnd_refine,y,x,indexing='ij')
    #
    (y_bnd_z,y_bnd_y,y_bnd_x)=np.meshgrid(z_refine,y_bnd,x,indexing='ij')
    #
    (x_bnd_z,x_bnd_y,x_bnd_x)=np.meshgrid(z_refine,y,x_bnd,indexing='ij')
    #

    
    #hole_min_z=2.5e-3; # 0.8e-3 
    ##hole_min_z=z_thick 
    #hole_min_x=3e-3
    #hole_max_x=20e-3
    #hole_min_y=12.e-3
    #hole_max_y=24.e-3
    #
    hole_k=0 # W/m/deg K
    hole_rho=rho # W/m/deg K
    hole_c=c # J/kg/deg K
    #

    # thininsulatinglayerdepth=2*dz
    # thininsulatinglayer_min_x=16e-3
    # thininsulatinglayer_max_x=32e-3
    # thininsulatinglayer_min_y=15e-3
    # thininsulatinglayer_max_y=30e-3
    

    materials=(
        # material 0: composite
        (heatsim2.TEMPERATURE_COMPUTE,np.diag(np.array((kz,ky,kx),dtype='d')),rho,c),
        #(heatsim2.TEMPERATURE_COMPUTE,kz,rho,c),
        # material 1: hole
        (heatsim2.TEMPERATURE_COMPUTE,np.diag(np.array((hole_k,hole_k,hole_k),dtype='d')),hole_rho,hole_c),
        #(heatsim2.TEMPERATURE_COMPUTE,hole_k,hole_rho,hole_c),
        # material 2: composite  (so we can use same material matrix as heatsim)
        (heatsim2.TEMPERATURE_COMPUTE,np.diag(np.array((kz,ky,kx),dtype='d')),rho,c),
        #(heatsim2.TEMPERATURE_COMPUTE,kz,rho,c),
        # material 3: Fixed temperature (Dirichlet boundary condition)
        (heatsim2.TEMPERATURE_FIXED,),
    )
    boundaries=(
        # boundary 0: conducting
        (heatsim2.boundary_conducting_anisotropic,),
        #(heatsim2.boundary_conducting,),
        # boundary 1: insulating
        (heatsim2.boundary_insulating,),
        # boundary 2: thininsulatinglayer
        # We want a thin insulating layer that halves 
        # the effective thermal effusivity of the remainder of the 
        # material 
        # e=sqrt(k rho c) has units of Joules/(m^2*deg K*sqrt(s))
        # effusivities are probably e1e2/(e1+e2) form
        # so we want e1e2/(e1+e2)=e2/2
        # e1/(e1+e2)=1/2
        # e1=e2
        # thin insulating layer coefficient is Joules/(m^2*deg K)
        # so we get this from effusivity * sqrt(characteristic time/2.0)
        # sqrt(kz*rho*c)*sqrt(thininsulatinglayerdepth**2/(2*np.pi*(kz/(rho*c))))
        # Simplifying: 
        #  rho*c*thininsulatinglayerdepth/np.sqrt(2.0*np.pi)
        (heatsim2.boundary_thininsulatinglayer,rho*c*thininsulatinglayerdepth/np.sqrt(2.0*np.pi))
    )
    volumetric=(  # on material grid
        # 0: nothing
        (heatsim2.NO_SOURCE,),
        #1: impulse source @ t=0
        (heatsim2.IMPULSE_SOURCE,0.0,flash_energy/(dz/dz_refinement)), # t (sec), Energy J/m^2
    )
    #
    # initialize all elements to zero
    (material_elements,
     boundary_z_elements,
     boundary_y_elements,
     boundary_x_elements,
     volumetric_elements)=heatsim2.zero_elements(nz_refine,ny,nx) 
    #
    # define nonzero material elements
    #
    material_elements[(zgrid_refine >= hole_min_z) &
                      (ygrid >= hole_min_y) &
                      (ygrid <= hole_max_y) &
                      (xgrid >= hole_min_x) &
                      (xgrid <= hole_max_x)]=1 # material 1: hole
    #
    volumetric_elements[0,:,:]=1  # set flash source (for heatsim2)
    #
    #
    # set edges to insulating
    boundary_x_elements[:,:,0]=1 # insulating
    boundary_x_elements[:,:,-1]=1 # insulating
    boundary_y_elements[:,0,:]=1 # insulating
    boundary_y_elements[:,-1,:]=1 # insulating
    boundary_z_elements[0,:,:]=1 # insulating
    boundary_z_elements[-1,:,:]=1 # insulating
    #
    

    # add thin insulating layer

    if thininsulatinglayerdepth < z_bnd[-1]:
        boundary_z_elements[ (z_bnd_x > thininsulatinglayer_min_x) &
                             (z_bnd_x < thininsulatinglayer_max_x) &
                             (z_bnd_y > thininsulatinglayer_min_y) &
                             (z_bnd_y < thininsulatinglayer_max_y) &
                             (z_bnd_z==z_bnd_refine[(np.abs(z_bnd_refine-thininsulatinglayerdepth)).argmin()])] = 2
        pass
    
    #
    # set boundaries of hole to insulating
    if hole_min_z < z_bnd[-1]:
        boundary_z_elements[ (z_bnd_x > hole_min_x) &
                             (z_bnd_x < hole_max_x) &
                             (z_bnd_y > hole_min_y) &
                             (z_bnd_y < hole_max_y) &
                             (z_bnd_z==z_bnd_refine[(np.abs(z_bnd_refine-hole_min_z)).argmin()])]=1 #
        pass
    #
    boundary_x_elements[ ((x_bnd_x == x_bnd[np.abs(x_bnd-hole_min_x).argmin()])|
                          (x_bnd_x == x_bnd[np.abs(x_bnd-hole_max_x).argmin()])) &
                         (x_bnd_y > hole_min_y) &
                         (x_bnd_y < hole_max_y) &
                         (x_bnd_z > hole_min_z)]=1
    #
    boundary_y_elements[ ((y_bnd_y == y_bnd[np.abs(y_bnd-hole_min_y).argmin()])|
                          (y_bnd_y == y_bnd[np.abs(y_bnd-hole_max_y).argmin()]))&
                         (y_bnd_x > hole_min_x) &
                         (y_bnd_x < hole_max_x) &
                        (y_bnd_z > hole_min_z)]=1
    #
    (ADI_params,ADI_steps)=heatsim2.setup(z[0],y[0],x[0],
                                          dz/dz_refinement,dy,dx,
                                          nz_refine,ny,nx,
                                          dt,
                                          materials,
                                          boundaries,
                                          volumetric,
                                          material_elements,
                                          boundary_z_elements,
                                          boundary_y_elements,
                                          boundary_x_elements,
                                          volumetric_elements)
    hs2trange=dt/2.0+np.arange(trange[-1]//dt+1)*dt
    #
    T=np.zeros((hs2trange.shape[0],ny,nx),dtype='d')
    fullstate=np.zeros((nz_refine,ny,nx),dtype='d')
    for tcnt in range(hs2trange.shape[0]):
        curt=hs2trange[tcnt]-dt/2.0
        print("t=%f" % (curt))
        fullstate=heatsim2.run_adi_steps(ADI_params,ADI_steps,curt,dt,fullstate,volumetric_elements,volumetric)
        T[tcnt,:,:]=fullstate[0,:,:] # extract top layer
        pass
    #
    return T[(T.shape[0]-trange.shape[0]):T.shape[0],:,:]
