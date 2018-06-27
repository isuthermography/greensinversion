import copy 
import numpy as np

# Tile a rectangle for inversion, with appropriate overlap
#

def build_tiled_rectangle(ny,nx,dy,dx,reflectors,wfmdata,xydownsample=1):
    # Warning: Results are in (y,x) order

    # Convert reflectors to numpy array
    reflectors=np.array(reflectors,dtype='d')
    # reflectors first column is z position
    # 2nd and third columns are reflector_ny and reflector_nx respectively
    
    # Convert input types
    nx=int(nx)
    ny=int(ny)
    dx=float(dx)
    dy=float(dy)

    tile_width=dx*nx
    tile_height=dy*ny
    
    # Width of uncertain boundary that is affected by 
    # stuff outside the tile.... based on the broadest 
    # reflector step size.... (perhaps should be based 
    # directly on lateral diffusivity???) 
    
    edgewidth=tile_width/4.0  # tile_width/np.min(reflectors[:,2])

    ## upper bound each edge at 1/3 of tile
    #if edgewidth > tile_width * (1.0/3.0):
    #    edgewidth = tile_width * (1.0/3.0)
    #    pass

    
    edgeheight=tile_height/4.0 # np.min(reflectors[:,1])
    #if edgeheight > tile_height * (1.0/3.0):
    #    edgeheight = tile_height * (1.0/3.0)
    #    pass

    centerradius_x=(tile_width/2.0 - edgewidth)
    centerradius_y=(tile_height/2.0 - edgeheight)

    minyminx_corners=[]
    yranges=[]
    xranges=[]
    
    yidx=0

    while yidx < wfmdata.shape[1]//xydownsample:
        ystep=int(round(ny-2.0*edgeheight/dy))
        assert(ystep > 0)

        if (yidx+ny >= wfmdata.shape[1]//xydownsample):
            # last y position... narrow it by starting early
            # but then step completely over it
            yidx = wfmdata.shape[1]//xydownsample-ny
            assert(yidx >= 0) # cannot operate on rectangle shorter than tile 
            ystep=ny
            pass
        
        xidx=0
        while xidx < wfmdata.shape[2]//xydownsample:
            xstep=int(round(nx-2.0*edgewidth/dx))
            assert(xstep > 0)
            
            if (xidx+nx >= wfmdata.shape[2]//xydownsample):
                # last x position... narrow it by starting early
                # but then step completely over it
                xidx = wfmdata.shape[2]//xydownsample-nx
                assert(xidx >= 0) # cannot operate on rectangle narrower than tile 
                xstep=nx
                pass
            
            minyminx_corners.append((yidx,xidx))
            yranges.append(np.arange(yidx,yidx+ny))
            xranges.append(np.arange(xidx,xidx+nx))
            
            xidx+=xstep
            pass
        
        yidx+=ystep
        pass
            
    # Compute contribution profiles
    # Contribution to a point comes from 
    # (sum of (values*weights))/(sum of weights)
    # weights come from hamming window centered inside 
    # border area and constant outside border area

    # determine weightmap from hamming window
    alpha=25.0/46.0  # hamming parameters
    beta=21.0/46.0   
    def hamm(Ry,Rx):
        # along the axis,
        # R = Rx/(4*centerradius) = 0.5 at the constant transition (cos==-1)
        # so Rx = 2*centerradius at the constant transition
        # Rx = centerradius at the steepest slope (cos==0)
        R=np.array(np.sqrt((Rx/(4.0*centerradius_x))**2.0+(Ry/(4.0*centerradius_y))**2.0),dtype='d')
        res=np.zeros(R.shape,dtype='d')
        res[R > 0.5]=alpha-beta # beyond central lobe treat as constant (alpha-beta)
        res[R <= 0.5]=alpha + beta*np.cos(2*np.pi*R[R <= 0.5])
        return res
    
    (offsety,offsetx)=np.meshgrid(dy*(np.arange(ny,dtype='d')-ny/2.0+0.5),dx*(np.arange(nx,dtype='d')-nx/2.0+0.5),indexing='ij')
    weightmap=hamm(offsety,offsetx)
    
    # Compute total weights over entire image
    totalweights=np.zeros((wfmdata.shape[1]//xydownsample,wfmdata.shape[2]//xydownsample),dtype='d')
    for tile_idx in range(len(minyminx_corners)):
        (yidx,xidx)=minyminx_corners[tile_idx]
        
        # print(weightmap.shape)
        # print(totalweights[yidx:(yidx+ny),xidx:(xidx+nx)].shape)
        # print(ny,nx)
        # accumulate weights
        totalweights[yidx:(yidx+ny),xidx:(xidx+nx)]+=weightmap
        pass

    
    
    # total results is (sum of (weights*values))/(sum of weights)
    # contributionprofiles entry is weight/(sum of weights)
    # Then to create total result for a pixel, we multiply contributionprofile
    # entry by result from that tile, and add it to similar products for
    # that pixel from all other tiles

    # compute contributionprofiles
    contributionprofiles=[]
    for tile_idx in range(len(minyminx_corners)):
        (yidx,xidx)=minyminx_corners[tile_idx]
        
        contributionprofiles.append(weightmap/totalweights[yidx:(yidx+ny),xidx:(xidx+nx)])
        pass

    return (minyminx_corners,yranges,xranges,contributionprofiles)
