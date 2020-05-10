import numpy as np
from scipy.ndimage import gaussian_filter


#for demon0
#get contour for disk
def get_contour(img,r=70,cent=(190,190),epsilon = 0.1):
    contour = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp = (i-cent[0])**2+(j-cent[1])**2
            if r-epsilon<np.sqrt(tmp)<r+epsilon: contour.append([i,j])
    return np.array(contour)

def loss_compute(img1,img2):
    ret = abs(img1-img2)
    return ret, np.sum(ret)

def get_inter(img1,img2):
    #get intersection, to get an idea of how much magnitude we should put
    pass

# scattered full grid in S we compute the gradient image of static first
def grad_image(img,direction="x"): 
    #convolutions with stride 1 with filter size 3x1 -> (-1,0,1)
    
    grad = np.zeros(img.shape)
    #convolution on horizontal
    if direction == "x":
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if x == 0: grad[y,x] = -img[y,x] + img[y,x+1]
                elif x == img.shape[1]-1: grad[y,x] - grad[y,x-1]
                else: grad[y,x] = img[y,x+1] - img[y,x-1]
    #convolution on vertical
    if direction == "y":
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if y == 0: grad[y,x] = -img[y,x] + img[y+1,x]
                elif y == img.shape[0]-1: grad[y,x] - grad[y-1,x]
                else: grad[y,x] = img[y+1,x] - img[y-1,x]
    return grad


#gaussian filter with thresh 0
#TODO

def gauss_filt():
	# regularize deformation field using Gaussian smoothing (but we should keep 0 if there is)
    # there shuould be no direction when intensity light is conserved
    # NEED TO ADAPT (FROM MY NOTEBOOK)
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


#for demon 2
def annisotrop_gauss():
	pass
