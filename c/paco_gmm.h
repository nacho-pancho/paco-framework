#ifndef PACO_GAUSSIAN_MIXTURE_MODEL_H
#define PACO_GAUSSIAN_MIXTURE_MODEL_H

#include "paco_types.h"

//============================================================================
struct paco_problem;

/**
 * matrix with concatenated centroids and inverse covariance matrices.
 * The number of modes is inferred from the matrix. For this, the argument must
 * be mxn where m is a multiple of n+1, so that the number of modes is m/n
 * each block of size (n+1) x n contains the mean of the model in the first row
 * and the inverse variance matrix in the following n rows.
 */
void paco_gmm_create ( const struct paco_problem* problem );

void paco_gmm_set_model ( gsl_matrix *model_data ) ;

void paco_gmm_set_noisy_patches ( const gsl_matrix *noisy ) ;

void paco_gmm_set_noise_sigma ( const sample_t sigma ) ;

void paco_gmm_destroy();

const gsl_vector *get_cluster_assignement ();

index_t paco_gmm_get_cluster_freq ( const index_t k );

index_t paco_gmm_get_num_clusters();

void paco_gmm_set_adapt_rate ( const sample_t rate );

void paco_gmm_fit ( const struct paco_problem* problem );

void paco_gmm_prox ( gsl_matrix *PB, const gsl_matrix *B, const double tau );

double paco_gmm_eval ( const gsl_matrix *B );

paco_function_st paco_gmm();

#endif