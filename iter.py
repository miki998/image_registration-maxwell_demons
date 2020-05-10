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


##################################################
#### Registration of source image, the grey anchor
##################################################
# Build a grid to recover position of each pixel of source image in final registered image
# Be careful : with inversion of A, you also need to invert x and y axis.
xim,yim = np.meshgrid(np.arange(0,nr),np.arange(0,nc))
xim = xim.reshape(1,-1)
yim = yim.reshape(1,-1)

#Coordinates of pixels in source image, corresponding to each point of the grid xim, yim

#######################
####Fill here

#since we have constant magnitude, we simply need to exert the same force 
#on all pixels in comparison to these contour
n_iter = 5
tmpIreca = Ireca.copy()
tmpIref = Iref.copy()
for i in range(1,n_iter+1):
    print('Iteration number {}'.format(i))
    
    gridRegInv = np.zeros((2,xim.shape[1]))
    
    force = 2
    d_fieldx, d_fieldy = comp_forces(tmpIreca[:,cutil][rutil],tmpIref[:,cutil][rutil]
                                     ,gIref_x,gIref_y,demons,force,epsilon = 0.001)
    
    break
    # regularize deformation field using Gaussian smoothing (but we should keep 0 if there is)
    # there shuould be no direction when intensity light is conserved
    nonzx = np.nonzero(d_fieldx != 0)
    nonzy = np.nonzero(d_fieldy != 0)

    maskx = np.zeros(d_fieldx.shape)
    masky = np.zeros(d_fieldy.shape)
    for x,y in zip(nonzx[0],nonzx[1]): maskx[x,y] = 1
    for x,y in zip(nonzy[0],nonzy[1]): masky[x,y] = 1


    d_fieldx = gaussian_filter(d_fieldx * maskx,sigma=1)
    d_fieldx /= gaussian_filter(maskx, sigma=1)
    d_fieldx[np.logical_not(maskx)] = 0
    
    d_fieldy = gaussian_filter(d_fieldy * masky,sigma=1)
    d_fieldy /= gaussian_filter(masky, sigma=1)
    d_fieldy[np.logical_not(masky)] = 0
    
    print(len(np.nonzero(d_fieldx)[0]))
    for idx in range(xim.shape[1]):
        x,y = xim[0][idx],yim[0][idx]
        gridRegInv[0,idx] = y - d_fieldx[y,x]
        gridRegInv[1,idx] = x - d_fieldy[y,x]
    
    ##############################
    ### Build registered image with source pixel associated
    Jreg = np.zeros((nr,nc))
    for i in range(xim.shape[1]):
        value_source = tmpIreca[int(round(gridRegInv[1,i]) + nr), int(round(gridRegInv[0,i]) + nc)]
        Jreg[xim[:,i],yim[:,i]] = value_source

    img_loss,loss = loss_compute(Jreg,Iref[:,cutil][rutil])
    print('This iteration loss: {}'.format(loss))

    #######################
    # display original images and affine transformation result
    ######################

    plt.figure(figsize=(15,15))

    plt.subplot(1,4,1)
    plt.imshow(Iref[:,cutil][rutil])
    #plt.imshow(Ireca[:,cutil][rutil])
    #plt.plot(Preca[0,PtsInd],Preca[1,PtsInd],'-ob',[Preca[0,PtsInd[2]],Preca[0,PtsInd[0]]],[Preca[1,PtsInd[2]],Preca[1,PtsInd[0]]],'-ob')
    #plt.plot(Pref[0,PtsInd],Pref[1,PtsInd],'-or',[Pref[0,PtsInd[2]],Pref[0,PtsInd[0]]],[Pref[1,PtsInd[2]],Pref[1,PtsInd[0]]],'-or')
    plt.title('Static Image')
    
    plt.subplot(1,4,2)
    plt.imshow(Ireca[:,cutil][rutil])
    plt.title('Original moving image')
    
    plt.subplot(1,4,3)
    #plt.imshow(tmpIref[:,cutil][rutil])
    plt.imshow(Jreg)
    #plt.plot(Pref[0,PtsInd],Pref[1,PtsInd],'sr','LineWidth',2)
    #plt.plot([Xreg1[0],Xreg2[0],Xreg3[0],Xreg1[0]],[Xreg1[1],Xreg2[1],Xreg3[1],Xreg1[1]],'-og','LineWidth',2)
    plt.title('After registration');
    
    plt.subplot(1,4,4)
    plt.imshow(img_loss)
    plt.title('Loss Image');
    plt.show()
    
    #update
    tmpIreca = np.concatenate((np.concatenate([pad,pad,pad],axis=1),
                            np.concatenate([pad,Jreg*255,pad],axis=1),
                            np.concatenate([pad,pad,pad],axis=1)),axis=0)