import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import scipy
from scipy.interpolate import griddata
from scipy.ndimage import convolve
import cv2
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib as mpl
from scipy.ndimage import gaussian_filter


# ### Implementation of demon 1
# - demons are on are scattererd on the the full grid of $S$ but where the intensity grad is non-zero.
# - free form deformation, effect of force using Gaussian filter
# - trilinear interpolation in $M$
# - magnitude of force given by optical flow, direction too, so then link back to trilinear interpolation to get final direction and momentum


#extract demons from S
demx = np.array((gIref_x!=0),dtype=int)
demy = np.array((gIref_y!=0),dtype=int)
#union of the coordinates of nonzero gradx or grady
demons = np.array(np.nonzero(demx + demy))

#computation of forces, for demon1
def comp_forces(Mimg,Simg,Xgrad,Ygrad,demons,force=1,epsilon=0.0001):
    #compute forces and directions of demons
    d_fieldx = np.zeros(Iref.shape)
    d_fieldy = np.zeros(Iref.shape)
    
    def comp_force(pts):
        x,y = pts
        m,s = Mimg[x,y], Simg[x,y]
        gradx, grady = Xgrad[x,y], Ygrad[x,y]
    
#        denomix = (gradx)**2+(m-s)**2
#        denomiy = (grady)**2+(m-s)**2
        denom = (grady)**2+(gradx)**2+(m-s)**2
    
        if denom < epsilon: directx = 0
        else: directx =(m-s)*gradx/denom
        
        if denom < epsilon: directy = 0
        else: directy =(m-s)*grady/denom
        
        return directx,directy

    for x,y in zip(demons[0],demons[1]):
        dx,dy = comp_force((x,y))
        
        tmp_norm = np.linalg.norm(np.array([dx,dy]))

        if tmp_norm == 0: 
            d_fieldx[x,y] = 0
            d_fieldy[x,y] = 0
        else:
            d_fieldx[x,y] = dx/tmp_norm
            d_fieldy[x,y] = dy/tmp_norm

    return force*d_fieldx,force*d_fieldy