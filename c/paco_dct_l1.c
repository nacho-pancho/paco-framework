#include <assert.h>
#include <math.h>
#include <string.h>
#include "paco_log.h"
#include "paco_problem.h"
#include "paco_dct.h"
#include "paco_dct_l1.h"
#include <omp.h>

//============================================================================

static gsl_vector *moe_betas = 0; ///< DCT coefficient weights
static char moe_remove_dc = 0;

//============================================================================

paco_function_st paco_dct_l1() {
    paco_function_st preset;
    preset.create = paco_dct_l1_create;
    preset.fit = paco_dct_l1_fit;
    preset.eval   = paco_dct_l1_eval;
    preset.prox   = paco_dct_l1_prox;
    preset.destroy = paco_dct_l1_destroy;
    strcpy ( preset.name, "wl1" );
    return preset;
}

//============================================================================

/**
 * weighted l1 cost function: f(X) = \sum_{i=1}^{n}\sum_{j=1}^{m} w_j |X_{ij} |
 */
double paco_dct_l1_eval ( const gsl_matrix *X ) {
    const index_t n = X->size1;
    const index_t m = X->size2;
    assert ( moe_betas->size == m );
    index_t NT;
#ifdef _OPENMP
#pragma omp parallel 
    NT = omp_get_num_threads();
#else
    NT = 1;
#endif
    gsl_vector* coeffs[NT];
    for (index_t T = 0; T < NT; T++) {
	    coeffs[T] = gsl_vector_alloc(m);
    } 
    double val = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:val)
#endif
    for ( index_t j = 0; j < n; j++ ) {
        const sample_t* pW = moe_betas->data;
#ifdef _OPENMP	
        const int T = omp_get_thread_num();
#else
        const int T = 1;
#endif
        gsl_vector* coeffsT = coeffs[T];
        gsl_vector_const_view Xjv = gsl_matrix_const_row ( X, j );
        if ( moe_remove_dc ) {
            gsl_vector_memcpy ( coeffsT, &Xjv.vector );
            paco_mapping_remove_dc_single ( coeffsT );
            dct2dn ( coeffsT, coeffsT );
        } else {
            dct2dn ( coeffsT, &Xjv.vector );
        }
        double valj = 0;
        const sample_t* pc = coeffsT->data;
        for ( index_t i = 0; i < m; ++i ) {
            valj += pW[i] * fabs ( pc[i] );
        }
        val += valj;
    }
    for (index_t T = 0; T < NT; T++) {
	    gsl_vector_free(coeffs[T]);
    } 

    return val / ( ( double ) m * ( double ) n ); // normalized
}

//============================================================================

/**
 * \brief Soft thresholding operator defined in EQUATION (11) in the paper.
 *
 * Given a matrix A and weights W
 * returns a matrix SA with the same dimensions as A in which each element is given by
 * SA_{i,j} = A_{i,j} - tau*W_{j} If A_{i,j} > tau
 *          = A_{i,j} + tau*W_{j} If A_{i,j} < -tau
 *          = SA_{i,j} = 0        Otherwise (-tau < A_{i,j} < tau)
 * @param  A   n x m input matrix,  W  1 x m weights, tau
 * @return SA  n x m matrix (preallocated)
 */

void paco_dct_l1_prox ( gsl_matrix *SA, const gsl_matrix *A, const double tau ) {
    assert ( SA->size1 == A->size1 );
    assert ( SA->size2 == A->size2 );
    const index_t m = A->size2;
    const index_t n = A->size1;
    const gsl_vector *W = moe_betas;
    if ( SA != A ) {
        gsl_matrix_memcpy ( SA, A );
    }
    dct2dn_batch ( SA, SA );
    const sample_t *pA = A->data;
    const size_t    sA = A->tda;
    sample_t *pSA = SA->data;
    const size_t    sSA = SA->tda;
    const sample_t *pW  = W->data;
    const sample_t tau2 = tau;

#ifdef _OPENMP
    #pragma omp parallel for shared(pW)
#endif
    for ( int i = 0; i < n; ++i ) {
        const sample_t *pAT = pA + i * sA;
        sample_t *pSAT = pSA + i * sSA;

        for ( int j = 0; j < m; ++j ) {
            const sample_t twj = tau2 * pW[j]; //gsl_vector_get(W,j);
            const sample_t Aij = pAT[j]; //gsl_matrix_get(A,i,j);

            if ( Aij > twj ) {
                pSAT[j] = Aij - twj; //gsl_matrix_set(SA, i, j, Aij - twj);
            } else if ( Aij < -twj ) {
                pSAT[j] = Aij + twj; //gsl_matrix_set(SA, i, j, Aij + twj);
            } else {
                pSAT[j] = 0; //gsl_matrix_set(SA, i, j, 0);
            }
        }
    }
    idct2dn_batch ( SA, SA );
}

//============================================================================


void paco_dct_l1_prox_vec ( gsl_vector *A, const gsl_vector *B, const double tau ) {
    const index_t m = A->size;
    const gsl_vector *W = moe_betas;

    sample_t *pA = A->data;
    const sample_t *pB = B->data;
    const sample_t *pW = W->data;

    for ( int i = 0; i < m; ++i ) {
        const sample_t ti = tau * pW[i];
        const sample_t Bi = pB[i];
        if ( Bi > ti ) {
            pA[i] = Bi - ti;
        } else if ( Bi < -ti ) {
            pA[i] = Bi + ti;
        } else {
            pA[i] = 0;
        }
    }
}

//============================================================================

void paco_dct_l1_create ( const struct paco_problem* problem ) {
    assert ( problem->mapping.patch_dim > 0 );
    moe_betas = gsl_vector_alloc ( problem->mapping.patch_dim );
    gsl_vector_set_all ( moe_betas, 1 );
    moe_betas->data[0] = 0;
    paco_dct_init ( problem->config.patch_width, problem->config.patch_width );
    moe_remove_dc = problem->config.remove_dc;
}

//============================================================================

void paco_dct_l1_fit ( const struct paco_problem* problem ) {
    const gsl_matrix *mask = problem->data.mask;
    const gsl_matrix* input = problem->data.input;
    const gsl_matrix* initial = problem->data.initial;
    const paco_config_st* cfg = &problem->config;
    const index_t w = cfg->patch_width;
    const index_t patch_dim = w * w;

    const index_t grid_height = input->size1 - w + 1;
    const index_t grid_width = input->size2 - w + 1;

    gsl_vector* coeffs = gsl_vector_alloc ( problem->mapping.patch_dim );

    gsl_vector_set_all ( moe_betas, 0 );
    sample_t *pw = moe_betas->data;
    index_t nsamples = 0;
    for ( index_t i = 0; i < grid_height; ++i ) {
        for ( index_t j = 0; j < grid_width; ++j ) {
            sample_t* pc = coeffs->data;
            if ( initial ) {
                paco_mapping_extract_single ( pc, initial, i, j, w, w, 1 );
                nsamples++;
            } else if ( is_patch_complete ( mask, i, j, w, w, 1 ) ) {
                paco_mapping_extract_single ( pc, input, i, j, w, w, 1 );
                nsamples++;
            } else { // incomplete patch: skip
                continue;
            }
            dct2dn ( coeffs, coeffs );
            for ( index_t k = 0; k < patch_dim; ++k ) {
                pw[k] += fabs ( pc[k] );
            }
        }
    }

    //
    // normalize so that largest weight is 1
    //
    sample_t minw = gsl_vector_min ( moe_betas );
    paco_info ( "num samples = %lu min weight = %f\n", nsamples, minw );
    paco_info ( "weights:\n" );
    for ( index_t j = 0; j < patch_dim; ++j ) {
        pw[j] = ( minw + 0.1 * nsamples ) / ( pw[j] + 0.1 * nsamples );
        paco_info ( "%7.5f\t", pw[j] );
    }
    pw[0] = 0;
    paco_info ( "\n" );
    paco_info ( "max weight = %f\n", gsl_vector_max ( moe_betas ) );

    //
    // free patches matrix
    //
    gsl_vector_free ( coeffs );
}

void paco_dct_l1_destroy() {
    gsl_vector_free ( moe_betas );
    paco_dct_destroy();
}
