/**
 * META-PACO
 * This mechanism adapts a restoration procedure which is not formulated as a variational problem
 * in that a restored patche y is a function (not the minimization of a function) of an original
 * patch z and possibly some parameters: y = f(z;theta)
 *
 * What we do is to solve for the perturbation in the *argument* of f so that consensus is achieve.
 * min_dj f(zj+dj;theta)  s.t. f(zj+d;theta) are in consensus
 *
 * The META-PACO problem is given by
 *
 * min_D ||f(Zj+Dj)-f(Zj)||^2 s.t. { f(Zj+Dj):j=1,\ldots,n} \in C
 *
 * and can be solved using ADMM on the splitting:
 *
 * min_D ||f(Zj+Dj)-f(Zj)||^2 + g(W) s.t. W=f(Zj+Dj)
 *
 * min_D ||f(Zj+Dj)-f(Zj)||^2 + g(W) s.t. W=f(Zj+Dj)
 *
 * an alternative is to seek for the smallest perturbation that achieves consensus:
 *
 * min_D ||Dj||^2  + g( f(D_j) )
 *
 * The solution to the above problems can be parameterized n the function f and its gradient:
 *
 *
 */
#ifndef PACO_META_H
#define PACO_META_H

#include "paco_types.h"

//============================================================================

/**
 * matrix with concatenated centroids and inverse covariance matrices.
 * The number of modes is inferred from the matrix. For this, the argument must
 * be mxn where m is a multiple of n+1, so that the number of modes is m/n
 * each block of size (n+1) x n contains the mean of the model in the first row
 * and the inverse variance matrix in the following n rows.
 */
void paco_gmm_set_model ( gsl_matrix *model_data ) ;

void paco_gmm_set_noisy_reference ( const gsl_matrix *noisy ) ;

void paco_gmm_set_noise_sigma ( const sample_t sigma ) ;

void paco_gmm_destroy();

const gsl_vector *get_clustering_map ();

index_t paco_gmm_get_cluster_freq ( const index_t k );

index_t paco_gmm_get_num_clusters();

void paco_gmm_set_adapt_rate ( const sample_t rate );

void paco_gmm_adapt_to ( gsl_matrix *X );


#endif