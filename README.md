## Algorithm

##### Notations
- $S$ is objective shape, $M$ is deformable model (can be rigid, non rigid/free etc...)
- $\phi$ is mapping

### Implementation of demon 0
- demons are on are scattererd on the contour of $S$.
- deformation are rigid, so one direction for all pixels
- iterative $\phi_n$ given by the affine transform, so explicit
- magnitude of force same, but correctness still assured since the number of support to exert force supposedly diminish when shapes overlap

DISCLAIMER: the maximum generality we allow ourselves is to deal only with disks of different direction, and allow ourselves only rigid transform plus white background

### Implementation of demon 1
- demons are on are scattererd on the the full grid of $S$ but where the intensity grad is non-zero.
- free form deformation, effect of force using Gaussian filter
- trilinear interpolation in $M$
- magnitude of force given by optical flow, direction too, so then link back to trilinear interpolation to get final direction and momentum
