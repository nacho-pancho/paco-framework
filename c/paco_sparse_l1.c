#include "paco_sparse_l1.h"
#include "paco_log.h"
#include <assert.h>
#include <math.h>
#include <string.h>
#include "paco_problem.h"

static gsl_vector *moe_betas = 0; ///< DCT coefficient weights

paco_function_st paco_sparse_l1() {
    paco_function_st preset;
    preset.create = paco_sparse_l1_create;
    preset.fit = paco_sparse_l1_fit;
    preset.eval   = paco_sparse_l1_eval;
    preset.prox   = paco_sparse_l1_prox;
    preset.destroy = paco_sparse_l1_destroy;
    strcpy ( preset.name, "sparsel1" );
    return preset;
}

//============================================================================

/**
 * sparse l1 cost function: f(X) = \sum_{i=1}^{n}\sum_{j=1}^{m} w_j |X_{ij} |
 */
double paco_sparse_l1_eval ( const gsl_matrix *X ) {
    //const gsl_vector *W = weights;
    //assert ( W->size == X->size2 );
    const index_t m = X->size1;
    const index_t n = X->size2;
    //const size_t    sX = X->tda;
    //
    // \todo: MUST EVALUATE ON DICT COEFFICIENTS, NOT ON X!
    //
    paco_error ( "PENDING: paco_sparse_l1_eval" );
    double val = 0;
    return val / m / n;
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

void paco_sparse_l1_prox ( gsl_matrix *SA, const gsl_matrix *A, const double tau ) {
    paco_error ( "PENDING: paco_sparse_l1_prox" );
}

//============================================================================

/**
 * \todo: do something for fuck's sake
 */
void paco_sparse_l1_create ( const struct paco_problem* problem ) {
    assert ( problem->mapping.patch_dim > 0 );
    moe_betas = gsl_vector_alloc ( problem->mapping.patch_dim );
    gsl_vector_set_all ( moe_betas, 1 );
    moe_betas->data[0] = 0;
}

//============================================================================

void paco_sparse_l1_fit ( const struct paco_problem* problem ) {
//
// \todo
//
    paco_error ( "PENDING: paco_Sparse_l1_fit" );
}

void paco_sparse_l1_destroy() {
    gsl_vector_free ( moe_betas );
}
