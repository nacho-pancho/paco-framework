#include <assert.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>

#include "paco_log.h"
#include "paco_problem.h"
#include "paco_l12.h"

//============================================================================

void paco_l12_create ( const struct paco_problem* problem ) {
    assert ( problem->mapping.patch_dim > 0 );
}

//============================================================================

void paco_l12_fit ( const struct paco_problem* problem ) {
    // no fitting here
}



//============================================================================

paco_function_st paco_l12() {
    paco_function_st preset;
    preset.create = paco_l12_create;
    preset.fit = paco_l12_fit;
    preset.eval   = paco_l12_eval;
    preset.prox   = paco_l12_prox;
    preset.destroy = paco_l12_destroy;
    strcpy ( preset.name, "l12" );
    return preset;
}

//============================================================================
/*
 * f(x) = \sum_{i=1}^{m} f_i( x_{2i-1}, x_{2i} ) = \sum_{i=1}^{m} \sqrt{ x_{2i-1}^2 + x_{2i}^2 }
 */
double paco_l12_eval ( const gsl_matrix *X ) {
    const index_t n = X->size1;
    const index_t m = X->size2;
    const index_t mn = m*n;
    double val = 0;
    const sample_t* px = X->data;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:val)
    #endif
    for ( index_t i = 0; i < mn; i += 2 ) { 
        val += sqrt(px[i] * px[i] + px[i+1] * px[i+1]);
    }
    return val / ( ( double ) m * ( double ) n ); // normalized
}

//============================================================================

/**
 *
 * proximal operator for the TV L1 cost function
 * ||x||_l121 = \sum_i ||(u_i,v_i)||_2 = \sum_i sqrt(u_i^2 + v_i^2)
 * where u_i = (Dh*x)_i
 *       v_i = (Dv*x)_i
 *
 * Dh is the horizontal gradient operator
 * Dv is the vertical gradient operator
 *
 * we solve the following problem.
 *
 * prox_{\mu l12}(y) = arg min_x \sum_i ||(u_i,v_i)||_2 + 1/2tau ||x - y||^2_2
 *
 * using ADMM and the following splitting
 *
 * prox_{\mu l12}(y) = arg_z min_(x,z) \sum_i||(z_2i,z_{2i+1}||_2 + 1/2tau ||x - y||^2_2 + 1/2mu ||Dx - z||^2_2 s.t Dx = z
 * where Dx = [ u_1, v_1, u_2, v_2, ... ]
 *
 * THIS IS WRONG BUT WORKS
 * The ADMM splittin cannot be Dx=z, it is guaranteed to work if the splitting is x=z
 * The Linearized ADMM (Uzawa's) should be used instead.
 * 
 * @param  A   n x m input matrix,  W  1 x m weights, tau
 * @return SA  n x m matrix (preallocated)
 */

void paco_l12_prox ( gsl_matrix *SA, const gsl_matrix *A, const double tau ) {
    assert ( SA->size1 == A->size1 );
    assert ( SA->size2 == A->size2 );
    const index_t m = A->size2;
    const index_t n = A->size1;
    if ( SA != A ) {
        gsl_matrix_memcpy ( SA, A );
    }

    const index_t mn = m*n;
    sample_t* pa = SA->data;
    //#ifdef _OPENMP
    //#pragma omp parallel for 
    //#endif
    for ( index_t i = 0; i < mn; i += 2 ) { 
        const double b = sqrt(pa[i] * pa[i] + pa[i+1] * pa[i+1]);
        if (b <= tau) {
            pa[i] = pa[i+1] = 0;
        } else {
            const double shrink = 1.0-tau/b;
            pa[i]   *= shrink;
            pa[i+1] *= shrink;
        }
    }
}

void paco_l12_destroy() {}

