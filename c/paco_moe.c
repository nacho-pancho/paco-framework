#include <assert.h>
#include <math.h>
#include <string.h>
#include "paco_log.h"
#include "paco_problem.h"
#include "paco_dct.h"
#include "paco_moe.h"
#include <omp.h>

//============================================================================

//static gsl_vector *moe_betas = 0; ///< DCT coefficient weights
static char moe_remove_dc = 0;
static sample_t moe_kappa = 2.0; /// default shape parameter. 
static sample_t moe_beta = 0.05; /// default scale parameter. 
static sample_t moe_normalization_term = 0;
static index_t moe_reweighting_iterations = 2;

//============================================================================

paco_function_st paco_moe() {
    paco_function_st preset;
    preset.create = paco_moe_create;
    preset.fit = paco_moe_fit;
    preset.eval   = paco_moe_eval;
    preset.prox   = paco_moe_prox;
    preset.destroy = paco_moe_destroy;
    strcpy ( preset.name, "moe" );
    return preset;
}



//============================================================================

/**
 * MOE cost function: f(X) = \sum_{i=1}^{n}\sum_{j=1}^{m}  (kappa+1) log( |X_{ij}| + beta_j) |
 */
double paco_moe_eval ( const gsl_matrix *X ) {
    const index_t n = X->size1;
    const index_t m = X->size2;
    //assert ( moe_betas->size == m );
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
#ifdef _OPENMP	    
        const int T = omp_get_thread_num();
#else	
        const int T = 1;
#endif	
        gsl_vector* coeffsT = coeffs[T];
	const sample_t beta = moe_beta;
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
            valj +=  log ( fabs ( pc[i] ) + beta );
        }
        val += valj;
    }
    val *= moe_kappa + 1.0;
    for (index_t T = 0; T < NT; T++) {
	    gsl_vector_free(coeffs[T]);
    } 

    return val / ( ( double ) m * ( double ) n ); // normalized
}

//============================================================================

/**
 * \brief approximate MOE proximal operator using reweighted L1
 * 
 * Each iteration linearizes the MOE prior around the previous iterate:
 * \hat{f}(a(k)) = (d/da)\phi( a(k) ; a(k-1),beta )*|a(k)|
 *  
 * (d/da)\phi(a(k); (k-1),beta)   = (d/da){ (kappa+1) log( |a| + beta) _|a=a(k-1) 
 *                                = (kappa+1)\frac{1}{ |a(k-1)| + beta }
 * \hat{f}(a(k)) = (kappa+1)\frac{1}{ |a(k-1)| + beta } |a(k)|
 * 
 * so the weight at each iteration is given by
 * w(k) = tau * (kappa + 1) / ( |a(k-1)| + beta)
 * 
 * we use a(0) = 0 as a starting point, so that
 * 
 * w(0) = tau * (kappa + 1)  / beta
 *  
 * @param  A   n x m input matrix,  W  1 x m weights, tau
 * @return SA  n x m matrix (preallocated)
 */

void paco_moe_prox ( gsl_matrix *SA, const gsl_matrix *A, const double tau ) {
    assert ( SA->size1 == A->size1 );
    assert ( SA->size2 == A->size2 );
    const index_t m = A->size2;
    const index_t n = A->size1;
    //const gsl_vector *W = moe_betas;
    if ( SA != A ) {
        gsl_matrix_memcpy ( SA, A );
    }
    dct2dn_batch ( SA, SA );
    const sample_t *pA = A->data;
    const size_t    sA = A->tda;
    sample_t *pSA = SA->data;
    const size_t    sSA = SA->tda;
    //const sample_t *pW  = W->data;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for ( int i = 0; i < n; ++i ) {
        const sample_t *pAT = pA + i * sA;
        sample_t *pSAT = pSA + i * sSA;

        for ( int j = 0; j < m; ++j ) {
            const sample_t betaj = moe_beta;
            //const sample_t betaj = pW[j];
            sample_t twj = tau * moe_normalization_term * (moe_kappa + 1.0) / betaj;
            sample_t Aij = pAT[j];

            for ( int k = 0; k < moe_reweighting_iterations; ++k ) {
                if ( Aij > twj ) {
                    Aij = Aij - twj; 
                } else if ( Aij < -twj ) {
                    Aij = Aij + twj; 
                } else {
                    Aij = 0; 
                }
                // weight at next iteration: w(k) = tau * (kappa + 1) / ( |a(k-1)| + beta)
                twj = tau * (moe_kappa+1) / ( fabs ( Aij ) + betaj );
            }

            pSAT[j] = Aij;
        }
    }
    idct2dn_batch ( SA, SA );
}

//============================================================================

/**
 * set DCT coefficient weights for weightef L1 cost function
 */
void paco_moe_create ( const paco_problem_st* problem ) {
    assert ( problem->mapping.patch_dim > 0 );
    //moe_betas = gsl_vector_alloc ( problem->mapping.patch_dim );
    //gsl_vector_set_all ( moe_betas, 1.0 );
    //moe_betas->data[0] = 0;
    paco_dct_init ( problem->config.patch_width, problem->config.patch_width );
    moe_remove_dc = problem->config.remove_dc;
    moe_kappa     = problem->config.moe_kappa;
    moe_reweighting_iterations = problem->config.moe_iter;
}

//============================================================================

void paco_moe_fit ( const struct paco_problem* problem ) {
    const gsl_matrix *mask = problem->data.mask;
    const gsl_matrix* input = problem->data.input;
    const gsl_matrix* initial = problem->data.initial;
    const paco_config_st* cfg = &problem->config;
    const index_t w = cfg->patch_width;
    const index_t patch_dim = w * w;

    const index_t grid_height = input->size1 - w + 1;
    const index_t grid_width = input->size2 - w + 1;

    gsl_vector* coeffs = gsl_vector_alloc ( problem->mapping.patch_dim );

    //gsl_vector_set_all ( moe_betas, 0 );
    //sample_t *pw = moe_betas->data;
    index_t n = 0;
    sample_t a = 0;
    for ( index_t i = 0; i < grid_height; ++i ) {
        for ( index_t j = 0; j < grid_width; ++j ) {
            sample_t* pc = coeffs->data;
            if ( initial ) {
                paco_mapping_extract_single ( pc, initial, i, j, w, w, 1 );
                n++;
            } else if ( is_patch_complete ( mask, i, j, w, w, 1 ) ) {
                paco_mapping_extract_single ( pc, input, i, j, w, w, 1 );
                n++;
            } else { // incomplete patch: skip
                continue;
            }
            dct2dn ( coeffs, coeffs );
            for ( index_t k = 0; k < patch_dim; ++k ) {
                //pw[k] += fabs ( pc[k] );
                a += fabs ( pc[k] );
            }
        }
    }
    moe_beta = a / ((double) n * (double) patch_dim);
    paco_info ( "MOE parameter beta: %f\n",moe_beta );
    //for ( index_t j = 0; j < patch_dim; ++j ) {
    //    pw[j] = (moe_kappa-1.0) * pw[j] / ( double ) n;
    //   paco_info ( "%7.5f\t", pw[j] );    
    //}
    //
    // in order to prevent ill-conditioning, we have to take
    // into account the relative weight of this prior to the
    // rest of the cost function.
    // as MOE is iterative, the weight varies, but it is always
    // equal or below than the one used at the first reweighted L1
    // iteration, which is (kappa+1)/beta
    //
    // therefore, we scale the weights so that, during the first
    // iteration, the maximum weight is 1
    //
    // as the "weights" matrix in this case contains the parameters "beta",
    // which are inversely proportional to the weights, we have to *divide* beta
    // in order to obtain a normalization
    //
    const sample_t max_weight = (moe_kappa+1.0)/moe_beta;
    moe_normalization_term = 1.0/max_weight;
    paco_info ( "\nMOE normalization term: %f\n",moe_normalization_term );
    //
    //
    // free patches matrix
    //
    gsl_vector_free ( coeffs );
}

void paco_moe_destroy() {
    //if ( moe_betas )
    //    free ( moe_betas );
    paco_dct_destroy();
}
