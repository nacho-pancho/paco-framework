# Stripped down PACO for public usage

This repository contains a basic PACO implementation which can be used to implement any patch-based restoration method which can be expressed as the minimization of a per-patch cost function subject to patch consensus constraints. 

In order to create your own PACO-based solution, you can do any of the two:
- copy and modify the functions in the PACO class
- inherit from PACO and overwrite the main functions: prox_f, prox_g, and init.

The code includes a small demo which implements the PACO-DCT inpainting method described in (citation needed) and IPOL http://ipol.im


