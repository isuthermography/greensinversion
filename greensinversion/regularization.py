import numpy as np



    

def apply_tikhonov_regularization(u,s,v,usetikparam,vector):
    #alpha = usetikparam*np.sqrt(u.shape[0]/v.shape[1]) # Tikhonov parameter interpreted as scaled by sqrt(matrix rows/matrix cols) so that it is directly interpretable as NETD/NESI  (noise equivalent temperature difference over noise equivalent source intensity, with NETD measured in deg. K and NESI measured in J/m^2

    # NOTE: u and v no longer orthogonal as they have already been pre-multiplied by scaling factors
    
    # tikhonov scaling temporarily disabled
    alpha=usetikparam

    d = s/(s**2+alpha**2) 
    #inverse = np.dot(v.T*(d.reshape(1,d.shape[0])),u.T)
    #return inverse
    return np.dot(v.T,np.dot(u.T,vector)*d)
    
