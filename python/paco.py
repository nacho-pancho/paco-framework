#
#-----------------------------------------------------------------------
# GENERIC PACO IMPLEMENTATION
#-----------------------------------------------------------------------
#
import time
import numpy as np
import numpy.linalg as la
import patches

#-----------------------------------------------------------------------

class PACO:
    """
    Generic, ADMM PACO implementation (over regular patch grids)
    """
    def __init__(self,input_signal,patch_shape,patch_stride):
        self.mapper        = patches.PatchMapper(input_signal.shape,patch_shape,patch_stride)
        if self.mapper.padded_shape != self.mapper.signal_shape:
            pad_size = [ (0, a-b) for (a,b) in zip(self.mapper.padded_shape,self.mapper.signal_shape)]
            self.input_signal  = np.pad(input_signal,pad_size,mode='edge')
        else:
            self.input_signal = input_signal
        self.aux_signal    = np.empty(self.mapper.padded_shape)
        self.output_signal = np.zeros(self.mapper.padded_shape)
        self.A             = np.zeros(self.mapper.patch_matrix_shape)
        self.B             = np.zeros(self.mapper.patch_matrix_shape)
        self.U             = np.zeros(self.mapper.patch_matrix_shape)
        #
        # auxiliary, for default stopping condition based on change in argument
        #
        self.prevB         = np.zeros(self.mapper.patch_matrix_shape)
        self.iter = 0

        
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
        px[np.abs(x) < tau] = 0
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
        return self.mapper.extract(self.aux_signal,px)

    
    def init(self):
        """
        Problem-specific initialization
        """
        self.A[:] = 0
        self.B[:] = 0
        self.U[:] = 0
        self.prevB[:] = 0
        self.iter = 0

    def monitor(self):
        pass

    def run(self, tau=1, maxiter=100, minchange=1e-5, check_every=1):
        tau *= np.max(self.input_signal)
        self.prevB = np.empty(self.mapper.patch_matrix_shape)
        #
        # ADMM ITERATION
        #
        tic = time.time()
        self.iter = 0
        while self.iter < maxiter:
            #
            # 0. advance iteration, copy previous state
            #
            self.iter += 1
            #
            # 1. proximal operator on A
            #
            self.prox_f(self.B - self.U, tau, self.A)
            #
            # 2. proximal operator on B
            #
            self.prox_g(self.A + self.U, tau, self.B)
            #
            # 3. update multiplier
            #
            self.U += self.A
            self.U -= self.B
            #
            # 4. check progress:
            #
            if (self.iter % check_every) == 0:
                self.monitor()
                #
                # 4.2 default stopping condition
                #
                rel_ch = np.linalg.norm(self.B - self.prevB,'fro') / ( np.linalg.norm(self.B,'fro') + 1e-5 )
                if rel_ch < minchange: 
                    break
                dt = time.time() - tic
                tic = time.time()
                print('iter',self.iter,'time',dt,'rel. chg.',rel_ch,'min. chg.',minchange)
                #
                # 4.3 save intermediate result
                #
                # self.mapper.stitch(self.A, self.output_signal)
                # (insert save statement here)
                np.copyto(self.prevB,self.B)
                #
                # end check progress
                #
            #
            # end main ADMM loop
            #
        self.mapper.stitch(self.B, self.output_signal)
        #
        # end run
        #
        return self.output_signal


class LinearizedPACO(PACO):
    """
    Linearized ADMM PACO implementation (over regular patch grids) for
    linear patch models based on a generic non-orthogonal dictionary
    """
    def __init__(self,input_signal,patch_shape,patch_stride, dictionary):
        super().__init__(input_signal,patch_shape,patch_stride)
        self.dictionary = dictionary

    def run(self, tau=1, mu=0.9,maxiter=100, minchange=1e-5, check_every=1):
        tau *= np.max(self.input_signal)
        self.prevB = np.empty(self.mapper.patch_matrix_shape)
        #
        # ADMM ITERATION
        #
        tic = time.time()
        self.iter = 0
        D = self.dictionary # short alias
        AD = np.dot(self.A, self.D)
        while iter < maxiter:
            # 1 proximal operator on A
            self.prox_f(self.A - (mu/tau)*np.dot((AD - self.B + self.U),self.D.T), mu, self.A)
            AD = np.dot(self.A,self.D)
            self.prox_g( AD + self.U, self.B )
            self.U += AD
            self.U -= self.B
            if (self.iter % check_every) == 0:
                self.monitor()
                #
                # 4.2 default stopping condition
                #
                rel_ch = np.linalg.norm(self.B - self.prevB,'fro') / ( np.linalg.norm(self.B,'fro') + 1e-5 )
                if rel_ch < minchange:
                    break
                dt = time.time() - tic
                tic = time.time()
                print('iter',self.iter,'time',dt,'rel. chg.',rel_ch,'min. chg.',minchange)
                #
                # 4.3 save intermediate result
                #
                # self.mapper.stitch(self.A, self.output_signal)
                # (insert save statement here)
                np.copyto(self.prevB,self.B)
                #
                # end check progress
                #
            #
            # end main ADMM loop
            #
            self.mapper.stitch(self.B, self.output_signal)
