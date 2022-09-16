# Stripped down PACO for public usage

This repository contains a basic PACO implementation which can be used to implement any patch-based restoration method which can be expressed as the minimization of a per-patch cost function subject to patch consensus constraints. 

In order to create your own PACO-based solution, you can do any of the two:
- copy and modify the functions in the PACO class
- inherit from PACO and overwrite the main functions: prox_f, prox_g, and init.

The code includes a small demo which implements the PACO-DCT inpainting method described in (citation needed) and IPOL http://ipol.im

## Performance

The default implementation and, in particular, the patch extraction/stitching implementation, is **extremely** slow, as it
is generic for N-D signals.

Orders of magnitude faster implementations can be obtained simply by:
1. writing specialized versions of extract/stitch (1D, 2D, etc.) in Python (that's already a huge advantage)
1. writing such things in C through the Python C API
1. writing such things in some parallel/GPU based package such as JAX
