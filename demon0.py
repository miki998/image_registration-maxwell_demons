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
from helpers import *

### Implementation of demon 0
# - demons are on are scattererd on the contour of $S$.
# - deformation are rigid, so one direction for all pixels
# - iterative $\phi_n$ given by the affine transform, so explicit
# - magnitude of force same, but correctness still assured since the number of support to exert force supposedly diminish when shapes overlap

# DISCLAIMER: the maximum generality we allow ourselves is to deal only with disks of different direction, and allow ourselves only rigid transform plus white background


#DEMO USE
#shape creation
W, H = 300,360
# Create a black image
img1 = np.zeros((W,H,3))+255
overlay = img1.copy()
# (2) draw shapes:
cv2.circle(img1, (125, 130), 70, (40, 40, 40), -1)
# (3) blend with the original:
#opacity = 0.2
#cv2.addWeighted(overlay, opacity, img1, 1 - opacity, 0, img1)

img2 = np.zeros((W,H,3))+255
cv2.circle(img2, (190, 190), 70, (150, 150, 150), -1)


Ireca = np.array(img1,dtype=int)
Iref = np.array(img2,dtype=int)
# zero-padding to avoid out-of-bounds problems when one moves images
nr,nc,_ = Ireca.shape

pad = 255*np.ones((nr,nc,3),dtype=np.int8);
Iref = np.concatenate((np.concatenate([pad,pad,pad],axis=1),np.concatenate([pad,Iref,pad],axis=1),np.concatenate([pad,pad,pad],axis=1)),axis=0)
Ireca = np.concatenate((np.concatenate([pad,pad,pad],axis=1),np.concatenate([pad,Ireca,pad],axis=1),np.concatenate([pad,pad,pad],axis=1)),axis=0)

# positions of "utils pixels"
rutil = np.arange(nr,(2*nr))
cutil = np.arange(nc,(2*nc))

# display original images
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.imshow(Iref[:,cutil,:][rutil])
plt.title('target image')
plt.subplot(1,3,2)
plt.imshow(Ireca[:,cutil,:][rutil])
plt.title('source image')
plt.subplot(1,3,3)
plt.imshow(Iref[:,cutil,:][rutil])
plt.imshow(Ireca[:,cutil,:][rutil],alpha=0.5)
plt.title('two images superimposed')
plt.show()

def demon0_iter(sImg,xim,yim,force):
    #sImg: source image
    #xim,yim meshgrid to know where to move
    #return the the modified Rxim,Ryim (here supposedly just same direction)
    #eImg: exact image
    
    #compute direction from each demons of target image
    direction = np.array([0,0])

    for x,y in Pref.T:
        #intersected with source image
        tocenter = np.array([x-190,y-190])

        if (sImg[:,cutil,:][rutil][x,y] == 255).all(): direction += tocenter
        else: direction -= tocenter

    if (direction == 0).all(): raise ValueError('Division by 0 u stupid')
    #normalize direction
    direction = direction/Pref.T.shape[0]
    return force*direction/np.linalg.norm(direction)