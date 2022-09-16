#
#-----------------------------------------------------------------------
# SAMPLE IMPLEMENTATION: INPAINTING
#-----------------------------------------------------------------------
#
import numpy as np
import pnm
from scipy import fft
import paco

#-----------------------------------------------------------------------

class PacoDctInpainting(paco.PACO):
    """
    Generic PACO implementation (over regular patch grids)
    """
    def __init__(self,input_signal,input_mask,patch_shape,patch_stride):
        super().__init__(input_signal,patch_shape,patch_stride)
        if self.mapper.padded_shape != self.mapper.signal_shape:
            pad_size = [ (0, a-b) for (a,b) in zip(self.mapper.padded_shape,self.mapper.signal_shape)]
            self.input_mask = np.pad(input_mask, pad_size)
        else:
            self.input_signal = input_mask


    def prox_f(self, x, tau, px = None):
        """
        Proximal operator of  f(x): z = arg min f(x) + tau/2||z-x||^2_2
        In PACO, f(x) is the main cost function to be minimized
        """
        if px is None:
            px = np.empty(self.mapper.patch_matrix_shape)
        #
        # sample: soft thresholding
        #
        np.copyto(px,x)
        # terribly slow:
        for i in range(self.mapper.num_patches):
            patch = np.reshape(px[i,:],patch_shape)
            fft.dctn(patch,norm="ortho",overwrite_x=True)
            patch[np.abs(patch) < tau] = 0
            fft.idctn(patch,norm="ortho",overwrite_x=True)
            px[i, :] = patch.ravel()
        return px


    def prox_g(self, x, tau, px=None):
        """
        Proximal operator of g(x): z = arg min g(x) + tau/2||z-x||^2_2
        In PACO, g(x) is usually the indicator function of the constraint set
        which again is usually the intersection between the consensus set and
        additional constraints imposed by the problem.
        
        The sample implementation below just projects onto the consensus set.
        """
        if px is None:
            px = np.empty(self.mapper.patch_matrix_shape)
    
        self.mapper.stitch(x, self.aux_signal)
        self.aux_signal[self.input_mask] = self.input_signal[self.input_mask] # inpainting constraint
        return self.mapper.extract(self.aux_signal,px)

    def monitor(self):
        print(np.min(self.aux_signal),np.max(self.aux_signal))
        out = np.minimum(255,np.maximum(0,self.aux_signal)).astype(np.uint8)
        pnm.imsave(f'inpainting_iter_{self.iter:04d}.pnm', out)
        pass

if __name__ == '__main__':
    print("PACO - DCT - INPAINTING")
    print("*** NOTE -- THIS IS SUPER SLOW ***")
    _ref_  = pnm.imread("../data/test_nacho.pgm").astype(np.double)
    _mask_ = pnm.imread("../data/test_nacho_mask.pbm").astype(bool)
    _input_= _ref_ * _mask_ # zero out data within mask
    pnm.imsave('inpainting_input_.pnm',_input_.astype(np.uint8))
    pnm.imsave('inpainting_mask_.pnm',_mask_.astype(np.uint8))
    patch_shape = (8,8)
    patch_stride = (2,2)
    paco = PacoDctInpainting(_input_,_mask_, patch_shape, patch_stride)
    paco.init()
    _output_ = paco.run(tau=0.5,check_every=5)
    pnm.imsave('inpainting_output_.pnm',np.maximum(0,np.minimum(255,_output_)).astype(np.uint8))
