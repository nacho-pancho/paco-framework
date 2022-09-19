# Stripped down PACO for public usage

This repository contains a basic PACO implementation which can be used to implement any patch-based restoration method which can be expressed as the minimization of a per-patch cost function subject to patch consensus constraints. 

There is a framework for C and another one for Python.

The one in Python is really a bare bones one, with just a few lines of code.

The one in C is much larger, but just because I included a lot of functions, solvers, proximal operators, priors, etc., that I 
have already used and that are handy. Also, they serve as an example.

## Python version

The Python version is really basic. It contains a class named PACO which has a couple of main functions that you need to overwrite and that's it.

In order to create your own PACO-based solution, you can do any of the two:
- copy and modify the functions in the PACO class
- inherit from PACO and overwrite the main functions: prox_f, prox_g, and init.

The code includes a small demo which implements the PACO-DCT inpainting method described in (citation needed) and IPOL http://ipol.im

### Performance of the Python version

The code is reasonably fast (for Python). The bottleneck is usually in the patch extraction/stitching step.

In any case, this can be greatly improved by, for example:
1. writing such things in C through the Python C API
1. writing such things in some parallel/GPU based package such as JAX

## C version

This one is really quite optimized. The extraction/stitching steps are much faster than the Python counterparts.
The rest not so much, as most of the operations/proximal operators can be efficiently written using matrix/vector operations
on which Python is really fast. If you add JAX and a GPU on top of that, Python will surely end up being faster.

Nevertheless, betware of the extraction/stitching step because that is usually the bottleneck.

Also, as said above, the C code is much larger and complex. The framework functionality is achieved using function prototypes
(pointers to functions, in C jargon). You need to create your own functions for the proximal operators, initialization, monitoring,
etc. Luckily, some of the functions may already be implemented for you. I'd love to hear comments on how to make this framework
easy and clear to use. It is documented, but that's not enough.


