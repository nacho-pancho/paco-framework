#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Generic, non-efficient patch extraction and stitching
#
import numpy as np


class PatchMapper:
    """
    Extract and stitch patches of any dimension.
    NOTE: this implementation is *painfully* slow. We recommend you to replace it with a fast (less generic) one
    """
    def __init__(self,signal_shape,patch_shape,patch_stride):
        if np.isscalar(signal_shape):
            self.signal_shape  = [signal_shape]
        else:
            self.signal_shape = signal_shape

        if np.isscalar(patch_shape):
            self.patch_shape  = [patch_shape]
        else:
            self.patch_shape = patch_shape

        if np.isscalar(patch_stride):
            self.patch_stride = [patch_stride]
        else:
            self.patch_stride = patch_stride
            
        self.grid_shape   = [ grid_size(l,w,s) for (l,w,s) in zip(self.signal_shape,self.patch_shape,self.patch_stride) ]
        self.num_patches  = np.prod(self.grid_shape)
        self.patch_dim    = np.prod(self.patch_shape)
        self.padded_shape   = [ padded_size(l,w,s) for (l,w,s) in zip(self.signal_shape,self.patch_shape,self.patch_stride) ]
        self.norm_term    = PatchMapper.compute_norm_term(self.padded_shape,self.patch_shape,self.patch_stride,self.grid_shape)
        self.patch_matrix_shape = (self.num_patches,self.patch_dim)

    
    def extract(self, x, y = None):
        """
        Extract patches.
        NOTE: this implementation is *painfully* slow. We recommend you to replace it with a fast (less generic) one
        """
        stride = np.array(self.patch_stride)

        if y is None:
            y = np.zeros(self.patch_matrix_shape)

        for i,gi in enumerate(np.ndindex(*self.grid_shape)): # iterate over patch grid
            offset = stride * np.array(gi)
            for j,rel_idx in enumerate(np.ndindex(*self.patch_shape)):
                abs_idx = np.array(rel_idx) + offset
                y[i,j] = x[tuple(abs_idx)].ravel()
        return y

    
    def stitch(self, y, x = None):
        """
        Stitch patches
        NOTE: this implementation is *painfully* slow. We recommend you to replace it with a fast (less generic) one
        """
        if x is None:
            x = np.empty(self.signal_shape)                           

        x[:] = 0

        stride = np.array(self.patch_stride)

        for i,gi in enumerate(np.ndindex(*self.grid_shape)): # iterate over patch grid
            offset = stride * np.array(gi)
            for j,rel_idx in enumerate(np.ndindex(*self.patch_shape)): # iterate within patch
                abs_idx = np.array(rel_idx) + offset
                x[tuple(abs_idx)] += y[i,j]
        
        x[:] *= self.norm_term

        return x

    @staticmethod
    def compute_norm_term(signal_shape, patch_shape, patch_stride, grid_shape):
        """
        auxiliary matrix for normalization used for stitching
        """
        norm_term = np.zeros(signal_shape)
        for i,gi in enumerate(np.ndindex(*grid_shape)): # iterate over patch grid
            offset = patch_stride * np.array(gi)
            for j,rel_idx in enumerate(np.ndindex(*patch_shape)): # iterate within patch
                abs_idx = np.array(rel_idx) + offset
                norm_term[tuple(abs_idx)] += 1
        return 1.0 / norm_term


class PatchMapper2D:
    """
    Extract and stitch patches / 2D version
    """

    def __init__(self, signal_shape, patch_shape, patch_stride):
        if np.isscalar(signal_shape):
            self.signal_shape = [signal_shape]
        else:
            self.signal_shape = signal_shape

        if np.isscalar(patch_shape):
            self.patch_shape = [patch_shape]
        else:
            self.patch_shape = patch_shape

        if np.isscalar(patch_stride):
            self.patch_stride = [patch_stride]
        else:
            self.patch_stride = patch_stride

        self.grid_shape = [grid_size(l, w, s) for (l, w, s) in
                           zip(self.signal_shape, self.patch_shape, self.patch_stride)]
        self.num_patches = np.prod(self.grid_shape)
        self.patch_dim = np.prod(self.patch_shape)
        self.padded_shape = [padded_size(l, w, s) for (l, w, s) in
                             zip(self.signal_shape, self.patch_shape, self.patch_stride)]
        self.norm_term = PatchMapper2D.compute_norm_term(self.padded_shape, self.patch_shape, self.patch_stride, self.grid_shape)
        self.patch_matrix_shape = (self.num_patches, self.patch_dim)

    def extract(self, x, y=None):
        """
        Extract patches
        """
        stride = np.array(self.patch_stride)
        w1,w2 = self.patch_shape
        if y is None:
            y = np.zeros(self.patch_matrix_shape)

        for k, gk in enumerate(np.ndindex(*self.grid_shape)):  # iterate over patch grid
            i,j = stride * np.array(gk)
            y[k,:] = x[i:i+w1,j:j+w2].ravel()
        return y

    def stitch(self, y, x=None):
        """
        Stitch patches
        """
        if x is None:
            x = np.empty(self.signal_shape)
        x[:] = 0
        stride = np.array(self.patch_stride)
        w1,w2 = self.patch_shape

        for k, gk in enumerate(np.ndindex(*self.grid_shape)):  # iterate over patch grid
            i,j = stride * np.array(gk)
            x[i:i+w1,j:j+w2] += np.reshape(y[k,:],self.patch_shape)

        x[:] *= self.norm_term

        return x

    @staticmethod
    def compute_norm_term(signal_shape, patch_shape, patch_stride, grid_shape):
        """
        auxiliary matrix for normalization used for stitching
        """
        norm_term = np.zeros(signal_shape)
        w1,w2 = patch_shape

        for k, gk in enumerate(np.ndindex(*grid_shape)):  # iterate over patch grid
            i,j = patch_stride * np.array(gk)
            norm_term[i:i+w1,j:j+w2] += 1
        return 1.0 / norm_term


def grid_size(l,w,s):
    """
    number of patches along a given direction
    """
    return int(np.ceil((l-w)/s) + 1)


    
def padded_size(l,w,s):
    """
    compute the padded image size for a given width and stride
    """
    g = grid_size(l,w,s)
    return (g-1)*s + w

        



