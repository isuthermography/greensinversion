import numpy as np

def build_gi_grid(dy,ly,dx,lx,firstcentery=None,firstcenterx=None):
    #z_bnd=np.arange(nz+1,dtype='d')*dz  # z boundary starts at zero
    #

    yoffset=0.0
    xoffset=0.0

    if firstcentery is not None:
        yoffset = firstcentery-dy/2.0
        pass

    if firstcenterx is not None:
        xoffset = firstcenterx-dx/2.0
        pass


    ny=int(round(ly/dy))
    nx=int(round(lx/dx))
    #
    y_bnd=np.arange(ny+1,dtype='d')*dy + yoffset
    x_bnd=np.arange(nx+1,dtype='d')*dx + xoffset
    #
    # Create x,y,z element center grid
    #z=z_bnd[:-1]+dz/2.0
    y=y_bnd[:-1]+dy/2.0
    x=x_bnd[:-1]+dx/2.0
    #
    # Create 3d meshgrids indicating z boundary location
    # for all x,y center positions
    #
    # Voxel at i, j, k is has x boundarys at x_bnd[i] and x_bnd[i+1],
    # centered at y[k],z[k]
    # Same voxel has y boundaries at y_bnd[j] and y_bnd[j+1] which
    # are centered at x=x[i] and z=z[k]
    #
    #
    # create 3d meshgrids indicating element centers
    # print z.shape,y.shape,x.shape
    (ygrid,xgrid) = np.meshgrid(y,x,indexing='ij')
    #
    return (ny,nx,
            y,x,
            ygrid,xgrid,
            y_bnd,x_bnd)



def build_gi_grid_3d(dz,nz,dy,ly,dx,lx,firstcentery=None,firstcenterx=None):
    z_bnd=np.arange(nz+1,dtype='d')*dz  # z boundary starts at zero
    

    yoffset=0.0
    xoffset=0.0

    if firstcentery is not None:
        yoffset = firstcentery-dy/2.0
        pass

    if firstcenterx is not None:
        xoffset = firstcenterx-dx/2.0
        pass


    ny=int(round(ly/dy))
    nx=int(round(lx/dx))
    #
    y_bnd=np.arange(ny+1,dtype='d')*dy + yoffset
    x_bnd=np.arange(nx+1,dtype='d')*dx + xoffset
    #
    # Create x,y,z element center grid
    z=z_bnd[:-1]+dz/2.0
    y=y_bnd[:-1]+dy/2.0
    x=x_bnd[:-1]+dx/2.0
    #
    # Create 3d meshgrids indicating z boundary location
    # for all x,y center positions
    #
    # Voxel at i, j, k is has x boundarys at x_bnd[i] and x_bnd[i+1],
    # centered at y[k],z[k]
    # Same voxel has y boundaries at y_bnd[j] and y_bnd[j+1] which
    # are centered at x=x[i] and z=z[k]
    #
    #
    # create 3d meshgrids indicating element centers
    # print z.shape,y.shape,x.shape
    (zgrid,ygrid,xgrid) = np.meshgrid(z,y,x,indexing='ij')
    #
    return (ny,nx,
            z,y,x,
            zgrid,ygrid,xgrid,
            z_bnd,y_bnd,x_bnd)

