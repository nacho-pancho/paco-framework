#include "paco_types.h"
#include "paco_log.h"
#include <assert.h>
#include <math.h>


//============================================================================

/**
 */
double paco_tv_eval ( const gsl_matrix *X ) {
    // \todo
    paco_error ( "PENDING: paco_tv_eval" );
    return 0;
}

//============================================================================


void paco_tv_prox ( gsl_matrix *SA, const gsl_matrix *A, const double tau ) {
    paco_error ( "PENDING: paco_tv_prox" );
    // PYTHON CODE:
// def cs_itv_prox(D,P,b,xref,y,xini=None):
//     #
//     # Solve arg min_x (1/2)||Dx - u||_2^2 + (1/2L) ||x - y||_2^2 s.t. Px=b
//     #
//     m = D.shape[1]
//     if xini is None:
//         x = np.zeros(m)
//     else:
//         x = np.copy(xini)
//     z = rng.normal(size=(2*m))
//     u = np.zeros(2*m)
//     prevx = np.zeros(m)
//     maxiter = 500
//     eps = 1e-10
//     tau = 0.1
//     for iter in range(maxiter):
//         #print("iter",iter,end=", ")
//         np.copyto(prevx,x)

//         x = solve_prox_f_prox( D, z-u, P, b, tau, y )
//         z = solve_prox_g( D.dot(x) + u, tau )
//         u = u + D.dot(x) - z

//         dif = np.linalg.norm(x-prevx)/(eps+np.linalg.norm(x))
//         err = np.linalg.norm(x-0)/np.linalg.norm(xref)
//         merr = np.linalg.norm(np.dot(P,x)-b)/np.linalg.norm(b)
//         #print("iter",iter,"dif",dif) #if dif < 1e-8:
//         if dif < 1e-4:
//             break
//         #if kappa < 1:
//         #    tau = tau*kappa
//     return x

}

//============================================================================


// def compute_differential_operators(nrows,ncols):
//     N = nrows*ncols
//     Dh = np.eye(N,N)
//     li = 0
//     for i in range(nrows):
//         for j in range(ncols):
//             if j > 0:
//                 Dh[li,i*ncols+(j-1)] = -1
//             li = li+1
//     Dv = np.eye(N,N)
//     li = 0
//     for i in range(nrows):
//         for j in range(ncols):
//             if i > 0:
//                 Dv[li,(i-1)*ncols+j] = -1
//             li = li+1
//     D = np.concatenate((Dh,Dv))
//     return D


// def solve_prox_g(z,tau):
//     N2 = len(z)
//     N = int(N2/2)
//     z = np.reshape(z,(2,N))
//     nz = np.sqrt(np.sum(z**2,axis=0))
//     tz = np.outer(np.ones((2,1)),np.maximum(0,1-tau/(nz+1e-10)))
//     w = np.reshape(tz*z,(2*N))
//     return w


// DtD  = None
// DtDi = None
// PDtDi = None
// PDtDiDt = None
// PDtDiPt = None
// PDtDiPti = None

// def solve_prox_f_prox(D,w,P,b,tau,y):
//     global DtD
//     global DtDi
//     global PDtDi
//     global PDtDiPt
//     global PDtDiPti
//     #
//     # Solve arg min_x (1/2)||Dx - u||_2^2 + (1/2L) ||x - y||_2^2 s.t. Px=b
//     #
//     #
//     Dtw  = np.dot(D.T,w)
//     if DtD is None:
//         DtD   = np.dot(D.T,D)
//         DtDi = np.linalg.inv(DtD+(1/tau)*np.eye(DtD.shape[0]))
//         PDtDi   = np.dot(P,DtDi)
//         PDtDiPt = np.dot(PDtDi,P.T)
//         PDtDiPti = np.linalg.inv(PDtDiPt)

//     PDtDiDtw = np.dot(PDtDi, Dtw + (1/tau)*y)
//     L = np.dot( PDtDiPti, PDtDiDtw - b )
//     Ptl = np.dot(P.T,L)
//     x = np.dot( DtDi, Dtw + (1/tau)*y - Ptl )
//     return x

// def cs_itv_prox(D,P,b,xref,y,xini=None):
//     #
//     # Solve arg min_x (1/2)||Dx - u||_2^2 + (1/2L) ||x - y||_2^2 s.t. Px=b
//     #
//     m = D.shape[1]
//     if xini is None:
//         x = np.zeros(m)
//     else:
//         x = np.copy(xini)
//     z = rng.normal(size=(2*m))
//     u = np.zeros(2*m)
//     prevx = np.zeros(m)
//     maxiter = 500
//     eps = 1e-10
//     tau = 0.1
//     for iter in range(maxiter):
//         #print("iter",iter,end=", ")
//         np.copyto(prevx,x)

//         x = solve_prox_f_prox( D, z-u, P, b, tau, y )
//         z = solve_prox_g( D.dot(x) + u, tau )
//         u = u + D.dot(x) - z

//         dif = np.linalg.norm(x-prevx)/(eps+np.linalg.norm(x))
//         err = np.linalg.norm(x-0)/np.linalg.norm(xref)
//         merr = np.linalg.norm(np.dot(P,x)-b)/np.linalg.norm(b)
//         #print("iter",iter,"dif",dif) #if dif < 1e-8:
//         if dif < 1e-4:
//             break
//         #if kappa < 1:
//         #    tau = tau*kappa
//     return x
